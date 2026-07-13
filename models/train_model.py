
import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt

from pykt.config import que_type_models
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cal_loss(model, ys, r, rshft, sm, preloss=[], shft_time_intervals=None, loss_config=None):
    model_name = model.model_name

    if model_name in ["fluckt", "fluckt_noforgetting", "fluckt_nodf", "fluckt_noe"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)

        if loss_config is None:
            loss_config = {'type': 'bce', 'alpha': 0.25, 'gamma': 2.0}

        loss_type = loss_config.get('type', 'bce')

        if loss_type == 'binary_focal':
            from .loss import binary_focal_loss
            alpha = loss_config.get('alpha', 0.25)
            gamma = loss_config.get('gamma', 2.0)

            y_clamped = torch.clamp(y, eps, 1 - eps)
            y_logits = torch.log(y_clamped / (1 - y_clamped))

            base_loss = binary_focal_loss(y_logits, t, alpha=alpha, gamma=gamma, reduction='none')
        else:
            base_loss = binary_cross_entropy(y.double(), t.double(), reduction='none')

        loss = weighted_loss + preloss[0]

    return loss


def model_forward(model, data, rel=None, loss_config=None):
    model_name = model.model_name

    dcur = data

    q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
    qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur[
        "shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)


    time_intervals = None
    shft_time_intervals = None


    if "time_intervals" in dcur:
        time_intervals = dcur["time_intervals"].to(device)
        shft_time_intervals = time_intervals[:, 1:] if time_intervals is not None else None

    ys, preloss = [], []
    cq = torch.cat((q[:, 0:1], qshft), dim=1)
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)

    if model_name in ["fluckt", "fluckt_noforgetting", "fluckt_nodf", "fluckt_noe"]:
        y, reg_loss = model(cc.long(), cr.long(), cq.long(), time_intervals)
        ys.append(y[:, 1:])
        preloss.append(reg_loss)

    loss = cal_loss(model, ys, r, rshft, sm, preloss, shft_time_intervals, loss_config)

    return loss, ys

def compute_valid_loss(model, valid_loader, loss_config=None):
    model.eval()
    valid_losses = []

    with torch.no_grad():
        for data in valid_loader:
            loss, _ = model_forward(model, data, loss_config=loss_config)
            valid_losses.append(loss.item())

    return np.mean(valid_losses)


def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None,
                test_window_loader=None, save_model=False, data_config=None, fold=None, loss_config=None):
    max_auc, best_epoch = 0, -1
    train_step = 0

    rel = None

    train_losses = []
    valid_losses = []
    epochs = []

    for i in range(1, num_epochs + 1):
        loss_mean = []
        print(f"Epoch {i}/{num_epochs}")
        progress_bar = tqdm(train_loader, desc=f"Training", ncols=100)

        for data in progress_bar:
            train_step += 1
            if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt"]:
                model.model.train()
            else:
                model.train()

            loss, _ = model_forward(model, data, loss_config=loss_config)

            opt.zero_grad()
            loss.backward()  # compute gradients
            opt.step()  # update model's parameters

            loss_mean.append(loss.detach().cpu().numpy())

            if train_step % 10 == 0:  
                current_loss = np.mean(loss_mean[-10:])
                progress_bar.set_postfix(loss=f"{current_loss:.4f}")


        epoch_train_loss = np.mean(loss_mean)

        print(f"Computing validation loss...")
        epoch_valid_loss = compute_valid_loss(model, valid_loader, loss_config)

        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)
        epochs.append(i)

        print(f"Evaluating on validation set...")
        auc, acc = evaluate(model, valid_loader, model.model_name)


        if auc > max_auc + 1e-3:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type + "_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name,
                                                              save_test_path)
            validauc, validacc = auc, acc
        print(
            f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {epoch_train_loss:.4f}, valid loss: {epoch_valid_loss:.4f}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(
            f"            testauc: {round(testauc, 4)}, testacc: {round(testacc, 4)}, window_testauc: {round(window_testauc, 4)}, window_testacc: {round(window_testacc, 4)}")

        if i - best_epoch >= 10:
            break

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, valid_losses, 'r-', label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    loss_curve_path = os.path.join(ckpt_path, 'loss_curve.png')
    plt.savefig(loss_curve_path)
    print(f"Loss curve saved to {loss_curve_path}")

    loss_data = {
        'epoch': epochs,
        'train_loss': train_losses,
        'valid_loss': valid_losses
    }
    loss_data_path = os.path.join(ckpt_path, 'loss_data.csv')
    pd.DataFrame(loss_data).to_csv(loss_data_path, index=False)
    print(f"Loss data saved to {loss_data_path}")

    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch
