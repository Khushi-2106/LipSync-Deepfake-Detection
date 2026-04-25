from validate import validate
from data import create_dataloader
from trainer.trainer import Trainer
from options.train_options import TrainOptions


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.data_label = "val"
    val_opt.real_list_path = "./datasets/AVLips/0_real"
    val_opt.fake_list_path = "./datasets/AVLips/1_fake"
    return val_opt


if __name__ == "__main__":
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    model = Trainer(opt)
    opt.fine_tune = True
    opt.pretrained_model = "checkpoints/ckpt.pth"

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    print("Length of data loader: %d" % (len(data_loader)))
    print("Length of val  loader: %d" % (len(val_loader)))

    patience = 5
    counter = 0

    best_acc = 0
    for epoch in range(opt.epoch):
        model.train()
        print("epoch: ", epoch + model.step_bias)
        for i, (img, crops, motion_maps, label, _) in enumerate(data_loader):
            model.total_steps += 1

            model.set_input((img, crops, motion_maps, label))
            model.forward()
            loss = model.get_loss()

            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(
                    "Train loss: {}\tstep: {}".format(
                        model.get_loss(), model.total_steps
                    )
                )

        
        model.eval()
        ap, fpr, fnr, acc = validate(model.model, val_loader, opt.gpu_ids)
        print(
            "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                epoch + model.step_bias, acc, ap, fpr, fnr
            )
        )
        if acc > best_acc:
           best_acc = acc
           counter = 0
           print(f"Saving best model at epoch {epoch}, acc={acc}")
           model.save_networks("best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

        print("Batch labels:", label)
        print("Num real:", (label == 0).sum().item(),
            "Num fake:", (label == 1).sum().item())
