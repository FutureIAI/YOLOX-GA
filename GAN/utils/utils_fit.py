import os

import torch
from tqdm import tqdm

from utils.utils import get_lr
import torch.nn as nn
from GAN.util import getMaskedPart,CoordinateScaling
import  numpy as np

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir,model_G_train,model_G,ema_G,optimizer_G,model_D_train,model_D,ema_D,optimizer_D,model_superNet,loss_history_D,loss_history_G,local_rank=0):
    loss        = 0
    val_loss    = 0

    # 训练阶段判别器和生成器的损失，重建损失
    train_lossValue_D_dark3 = 0
    train_lossValue_G_dark3 = 0
    train_lossValue_reConstruct_dark3 = 0

    # 验证阶段判别器和生成器的损失，重建损失
    val_lossValue_D_dark3 = 0
    val_lossValue_G_dark3 = 0
    val_lossValue_reConstruct_dark3 = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    model_D_train.train()
    model_G_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]


        half_imgs,half_boxs = batch[2], batch[3]

        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]


                half_imgs = half_imgs.cuda(local_rank)
                half_boxs = [ann.cuda(local_rank) for ann in half_boxs]

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        targets_dark3 = targets

        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            # outputs         = model_train(images)
            outputs         = model_train(half_imgs)



            _ = model_superNet(images) #
            real_sample_dark3 = _[0]
            input_G_dark3 = outputs[0]



            fake_sample_dark3 = model_G_train(input_G_dark3)

        # -------------------------------------------------------------------------#
        # 训练生成器和判别器
        # -------------------------------------------------------------------------#
            # #-----------------------#
            # 计算局部判别器的损失函数
            # #-----------------------#
            for p in model_D_train.parameters():
                p.requires_grad = True
                pass

            real_sample_out_dark3 = model_D_train(real_sample_dark3.detach())
            real_sample_out_dark3 = real_sample_out_dark3.mean()

            fake_sample_out_dark3 = model_D_train(fake_sample_dark3.detach())
            fake_sample_out_dark3 = fake_sample_out_dark3.mean()
            train_loss_D_dark3 = fake_sample_out_dark3 - real_sample_out_dark3
            train_lossValue_D_dark3 = train_lossValue_D_dark3 + train_loss_D_dark3.item()
            # optimizer_D.zero_grad()
            train_loss_D_dark3.backward(retain_graph=True)
            optimizer_D.step()
            # 截断
            for p in model_D_train.parameters():
                p.data.clamp_(-0.01, 0.01)
                pass

            # #-----------------------#
            # 计算生成器的损失函数
            # #-----------------------#
            for p in  model_D_train.parameters():
                p.requires_grad = False
                pass
            fake_sample_out_dark3 = model_D_train(fake_sample_dark3)
            train_loss_G_dark3 = -(fake_sample_out_dark3.mean())
            train_lossValue_G_dark3 = train_lossValue_G_dark3 + train_loss_G_dark3.item()
            # optimizer_G.zero_grad()
            train_loss_G_dark3.backward(retain_graph=True)

            # #-----------------------#
            # 计算重建损失
            # #-----------------------#
            lsloss = nn.MSELoss()
            # 将tensor格式的坐标转化为list
            bboxs = []
            for i in range(len(targets_dark3)):
                bboxs.append(targets_dark3[i].tolist())
                pass
            # 根据下采样的倍数把坐标进行缩放
            bboxs = CoordinateScaling(bboxs,scale=0.25)

            real_sample_copy_dark3 = real_sample_dark3.clone()
            fake_sample_copy_dark3 = fake_sample_dark3.clone()


            if len(bboxs[0]) != 0:
                real_sample_copy_dark3,fake_sample_copy_dark3 = getMaskedPart(real_sample_copy_dark3,fake_sample_copy_dark3,bboxs)
                train_loss_reConstruct_dark3 = lsloss(real_sample_copy_dark3.detach(),fake_sample_copy_dark3)
                train_lossValue_reConstruct_dark3 = train_lossValue_reConstruct_dark3 + train_loss_reConstruct_dark3.item()
                train_loss_reConstruct_dark3.backward()

                pass
            optimizer_G.step()
        # ----------------------------------------------------------------------------#
        # ----------------------------------------------------------------------------#
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value = yolo_loss(outputs, targets)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)
            pass
        if ema_G:
            ema_G.update(model_G_train)
            pass
        if ema_D:
            ema_D.update(model_D_train)


        # loss += loss_value.item()
        
        if local_rank == 0:
            # pbar.set_postfix(**{'loss'  : loss / (iteration + 1),
            #                     'lr'    : get_lr(optimizer)})
            # pbar.set_postfix(**{'loss': loss / (iteration + 1),
            #                     'lr': get_lr(optimizer)})
            pbar.set_postfix(**{'train_D_dark3_loss':train_lossValue_D_dark3/(iteration + 1),'train_G_dark3_loss':train_lossValue_G_dark3/(iteration + 1),'train_reConstruct_dark3_loss':train_lossValue_reConstruct_dark3/(iteration + 1)})

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        pass

    if ema_D:
        model_D_train_eval = ema_D.ema
    else:
        model_D_train_eval = model_D_train.eval()
        pass
    if ema_G:
        model_G_train_eval = ema_G.ema
    else:
        model_G_train_eval = model_G_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]


        half_imgs, half_boxs = batch[2], batch[3]

        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]


                half_imgs = half_imgs.cuda(local_rank)
                half_boxs = [ann.cuda(local_rank) for ann in half_boxs]
                pass
            targets_dark3 = targets

            # ----------------------#
            #   前向传播
            # ----------------------#
            # outputs         = model_train(images)
            outputs = model_train(half_imgs)


            _ = model_superNet(images)
            real_sample_dark3 = _[0]
            input_G_dark3 = outputs[0]
            fake_sample_dark3 = model_G_train(input_G_dark3)


        # ------------------------------------------------------------------#
        # 验证阶段Dark3
        # ------------------------------------------------------------------#

            real_sample_out_dark3 = model_D_train(real_sample_dark3.detach())
            real_sample_out_dark3 = real_sample_out_dark3.mean()
            fake_sample_out_dark3 = model_D_train(fake_sample_dark3.detach())
            fake_sample_out_dark3 = fake_sample_out_dark3.mean()
            val_loss_D_dark3 = fake_sample_out_dark3 - real_sample_out_dark3
            val_lossValue_D_dark3 = val_lossValue_D_dark3 + val_loss_D_dark3.item()
            fake_sample_out_dark3 = model_D_train(fake_sample_dark3.detach())
            val_loss_G_dark3 = -(fake_sample_out_dark3.mean())
            val_lossValue_G_dark3 = val_lossValue_G_dark3 + val_loss_G_dark3.item()


            # #-----------------------#
            # 计算重建损失
            # #-----------------------#
            lsloss = nn.MSELoss()
            # 将tensor格式的坐标转化为list
            bboxs = []
            for i in range(len(targets_dark3)):
                bboxs.append(targets_dark3[i].tolist())
                pass
            bboxs = CoordinateScaling(bboxs, scale=0.125)
            if len(bboxs[0]) != 0:
                real_sample_dark3, fake_sample_dark3 = getMaskedPart(real_sample_dark3, fake_sample_dark3, bboxs)
                val_loss_reConstruct_dark3 = lsloss(real_sample_dark3.detach(), fake_sample_dark3.detach())
                val_lossValue_reConstruct_dark3 = val_lossValue_reConstruct_dark3 + val_loss_reConstruct_dark3.item()
                # train_loss_reConstruct.backward()

                pass

        # ------------------------------------------------------------------#
        #-------------------------------------------------------------------#

        if local_rank == 0:
            # pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            # pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.set_postfix(**{'val_D_dark3_loss': val_lossValue_D_dark3 / (iteration + 1),
                                'val_G_dark3_loss': val_lossValue_G_dark3 / (iteration + 1),
                                'val_reConstruct_dark3_loss': val_lossValue_reConstruct_dark3 / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('train D dark3 Loss: %.3f || train G dark3 Loss: %.3f || train reConstruct dark3 Loss: %.3f || val D dark3 Loss: %.3f || val G dark3 Loss: %.3f || val reConstruct dark3 Loss: %.3f' % ( train_lossValue_D_dark3 / epoch_step,train_lossValue_G_dark3 / epoch_step,train_lossValue_reConstruct_dark3 / epoch_step, val_lossValue_D_dark3 / epoch_step_val,val_lossValue_G_dark3 / epoch_step_val,val_lossValue_reConstruct_dark3 / epoch_step_val))


        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#

        if ema_D:
            save_state_dict_D = ema_D.ema.state_dict()
        else:
            save_state_dict_D = model_D.state_dict()
            pass

        if ema_G:
            save_state_dict_G = ema_G.ema.state_dict()
        else:
            save_state_dict_G = model_G.state_dict()
            pass


        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            torch.save(save_state_dict_D, os.path.join(save_dir, "dark3-ep%03d-D.pth" % (epoch + 1)))
            torch.save(save_state_dict_G, os.path.join(save_dir, "dark3-ep%03d-G.pth" % (epoch + 1)))


        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            pass

            
        # torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))