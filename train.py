import os
import sys
import time
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from dataset import load_datasets
from losses import DiceLoss
from losses import FocalLoss
from losses import *
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou
import warnings
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')
device="cuda" if torch.cuda.is_available() else "cpu"
import matplotlib.pyplot as plt
def validate(model: Model, val_dataloader: DataLoader, epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            
            images, bboxes, gt_masks = data
            images = images.to(device)
            bboxes = [bbox.to(device) for bbox in bboxes]
            gt_masks = [mask.to(device) for mask in gt_masks]
            num_images = images.size(0)
            pred_masks, _ = model(images, bboxes)
            pred_masks = [mask.to(device) for mask in pred_masks]
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            print(
                f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )

    print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.model.state_dict()
    torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()


def train_sam(
    cfg: Box,
    model: Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    # test_loss = FocalTverskyLoss()

    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        # test_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        validated = False

        for iter, data in enumerate(train_dataloader):
            if epoch > 1 and epoch % cfg.eval_interval == 0 and not validated:
                validate(model, val_dataloader, epoch)
                validated = True

            data_time.update(time.time() - end)
            images, bboxes, gt_masks = data

            # Move tensors to device
            images = images.to(device)
            bboxes = [bbox.to(device) for bbox in bboxes]
            gt_masks = [mask.to(device) for mask in gt_masks]

            batch_size = images.size(0)

            optimizer.zero_grad()

            pred_masks, iou_predictions = model(images, bboxes)
            pred_masks = [mask.to(device) for mask in pred_masks]
            iou_predictions = [iou_pred.to(device) for iou_pred in iou_predictions]

            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)

            loss_focal = torch.tensor(0., device=device)
            loss_dice = torch.tensor(0., device=device)
            loss_iou = torch.tensor(0., device=device)
            # loss_test = torch.tensor(0., device=device, requires_grad=True)

            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks
                # loss_test += test_loss(pred_mask, gt_mask, num_masks)

            loss_total = 20. * loss_focal + loss_dice + loss_iou
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            # Update loss meters
            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            # test_losses.update(loss_test.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            # Print progress
            if (iter %50==0):
                print(batch_iou)
                print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                    f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                    f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                    f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                    f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                    f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                    # f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                    f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')
            torch.cuda.empty_cache()

            


def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main(cfg: Box) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    # Initialize model
    model = Model(cfg).to(device=device)
    model.setup()
    train_dataloader, val_dataloader = load_datasets(cfg, model.model.image_encoder.img_size)

    optimizer, scheduler = configure_opt(cfg, model)

    train_sam(cfg, model, optimizer, scheduler, train_dataloader, val_dataloader)
    validate(model, val_dataloader, epoch=0)


if __name__ == "__main__":
    main(cfg)
