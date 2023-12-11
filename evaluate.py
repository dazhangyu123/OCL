import os
import sys
sys.path.append(os.path.abspath('tools'))
from train_source import *
from utils.train_helper import get_model, modified_bn_forward, MetricLogger, flip
import torch.optim as optim
from utils.eval import build_eval_info
from copy import deepcopy

class Evaluater():
    def __init__(self, cuda=None, train_id=None, logger=None, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.method = self.method
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        self.current_MIoU = 0
        self.best_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.train_id = train_id
        self.logger = logger

        # set TensorboardX
        self.writer = SummaryWriter(self.checkpoint_dir)

        # Metric definition
        self.Eval = Eval(self.num_classes)

        # model
        self.model, params = get_model(self)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.model.eval()
        self.model.to(self.device)

        # load pretrained checkpoint
        if self.pretrained_ckpt_file is not None:
            path1 = os.path.join(*self.checkpoint_dir.split('/')[:-1], self.train_id + 'best.pth')
            path2 = self.pretrained_ckpt_file
            if os.path.exists(path1):
                pretrained_ckpt_file = path1
            elif os.path.exists(path2):
                pretrained_ckpt_file = path2
            else:
                raise AssertionError("no pretrained_ckpt_file")
            self.load_checkpoint(pretrained_ckpt_file)


        if args.prior > 0.0:
            assert isinstance(args.prior, float) and args.prior <= 1 and args.prior >= 0, 'False prior exists.'
            nn.BatchNorm2d.prior = None
            nn.BatchNorm2d.forward = modified_bn_forward
            nn.BatchNorm2d.prior = args.prior

        # dataloader
        self.dataloader = City_DataLoader(self) if self.dataset=="cityscapes" else GTA5_DataLoader(self)
        self.dataloader.val_loader = self.dataloader.data_loader
        self.dataloader.valid_iterations = min(self.dataloader.num_iterations, 500)
        self.epoch_num = ceil(self.iter_max / self.dataloader.num_iterations)

    def main(self):
        # choose cuda
        if self.cuda:
            current_device = torch.cuda.current_device()
            self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))
        else:
            self.logger.info("This model will run on CPU")

        if self.method == 'TTT':
            # validate
            self.TTT()
        elif self.method == 'baseline':
            self.validate()
        else:
            raise AssertionError("do not implement ttt method")

        self.writer.close()

    def TTT(self):
        self.logger.info('Test time training...')
        self.Eval.reset()

        anchor = deepcopy(self.model.state_dict())


        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.learning_rate, momentum=self.momentum,
                              weight_decay=self.weight_decay)


        metric_logger = MetricLogger(delimiter="  ")
        header = 'Adapt:'

        i = 0
        for (x, y, id) in metric_logger.log_every(self.dataloader.val_loader, 100, header):
            i += 1
            if self.cuda:
                x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

            x_flip = flip(x, -1)
            input = torch.cat([x, x_flip], dim=0)
            output_s = self.model(input)[0]
            output_s = torch.stack([output_s[0], flip(output_s[1], -1)], dim=0)


            output_s_norm = F.normalize(F.softmax(output_s, dim=1), p=2, dim=1)
            naug, c, h, w = output_s.shape
            output_s_ = output_s_norm[:,:,::self.downsampling,::self.downsampling].view(naug, c, -1)

            pos_loss = -(torch.mul(output_s_norm[0], output_s_norm[1])).sum(0).mean()
            neg_loss = ((output_s_[0].T @ output_s_[0]).mean() + (output_s_[1].T @ output_s_[1]).mean()) / naug

            loss = self.pos_coeff * pos_loss + self.neg_coeff * neg_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(loss=loss.item())
            metric_logger.meters['loss'].update(loss.item(), n=1)
            metric_logger.meters['loss-pos'].update(pos_loss.item(), n=1)
            metric_logger.meters['loss-neg'].update(neg_loss.item(), n=1)

            output_s = F.interpolate(output_s, size=x.size()[2:], mode='bilinear', align_corners=True)
            pred = output_s.mean(0, keepdim=True).data.cpu().numpy()

            label = y.cpu().numpy()
            argpred = np.argmax(pred, axis=1)

            self.Eval.add_batch(label, argpred)


            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.mask_ratio).float().to(self.device)
                        with torch.no_grad():
                            p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)


        val_info = build_eval_info(self.class_16, self.logger, self.current_epoch)
        PA, MPA, MIoU, FWIoU = val_info(self.Eval, "")

        self.Eval.Print_Every_class_Eval()

        return PA, MPA, MIoU, FWIoU

    @torch.no_grad()
    def validate(self):
        os.makedirs('./saved_images', exist_ok=True)
        self.logger.info('validating one epoch...')
        self.Eval.reset()
        MIous = []

        metric_logger = MetricLogger(delimiter="  ")
        header = 'Validation:'

        idx = 0

        for (x, y, id) in metric_logger.log_every(self.dataloader.val_loader, 100, header):
            idx += 1
            if self.cuda:
                x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

            # model
            pred = self.model(x)
            if isinstance(pred, tuple):
                pred_2 = pred[1]
                pred = pred[0]
            pred = F.interpolate(pred, size=x.size()[2:], mode='bilinear', align_corners=True)

            if self.flip:
                pred_P = F.softmax(pred, dim=1)

                x_flip = flip(x, -1)
                pred_flip = self.model(x_flip)
                if isinstance(pred_flip, tuple):
                    pred_flip = pred_flip[0]
                pred_flip = F.interpolate(pred_flip, size=x.size()[2:], mode='bilinear', align_corners=True)
                pred_P_flip = F.softmax(pred_flip, dim=1)
                pred_P_2 = flip(pred_P_flip, -1)
                pred_c = (pred_P+pred_P_2)/2
                pred = pred_c.data.cpu().numpy()
            else:
                pred = pred.data.cpu().numpy()
            label = torch.squeeze(y, 1).cpu().numpy()
            argpred = np.argmax(pred, axis=1)

            self.Eval.add_batch(label, argpred)

            MIous.append(self.Eval.Mean_Intersection_over_Union())


        val_info = build_eval_info(self.class_16, self.logger, self.current_epoch)
        PA, MPA, MIoU, FWIoU = val_info(self.Eval, "")

        self.Eval.Print_Every_class_Eval()

        return PA, MPA, MIoU, FWIoU


    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.module.load_state_dict(checkpoint)
            self.logger.info("Checkpoint loaded successfully from "+filename)

            if 'crop_size' in checkpoint:
                self.crop_size = checkpoint['crop_size']
                print(checkpoint['crop_size'], self.crop_size)
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(filename))


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser.add_argument('--source_dataset', default='None', type=str,
                            help='source dataset choice')
    arg_parser.add_argument('--city_name', default='None', type=str,
                            help='source dataset choice')
    arg_parser.add_argument('--flip', action='store_true',help="flip")

    # evaluation methods setting
    arg_parser.add_argument("--method", type=str, default='baseline', choices=['baseline', 'TTT'], help="Normalization mode")

    arg_parser.add_argument("--pos-coeff", type=float, default=3.0,
                        help='Positive loss coefficient')
    arg_parser.add_argument("--neg-coeff", type=float, default=1.0,
                        help='Variance regularization loss coefficient')
    arg_parser.add_argument("--mask-ratio", type=float, default=0.01,
                        help='masking ratio in the stochastic restoration')
    arg_parser.add_argument("--prior", type=float, default=0.0, help=
                        "the hyperparameter determine the weight of training statistic")
    arg_parser.add_argument("--downsampling", type=int, default=1, help=
                        "setting the downsampling level when calculating negative term occupies too large GPU memory")

    # optimizer
    arg_parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="initial learning rate for the segmentation network.")

    arg_parser.add_argument('--gpu', type=str, default="1",
                            help=" the num of gpu")

    args = arg_parser.parse_args()
    print(args)
    if args.split == "train": args.split = "val"
    if args.checkpoint_dir == "none": args.checkpoint_dir = args.pretrained_ckpt_file + "/eval"
    args, train_id, logger = init_args(args)
    args.batch_size_per_gpu = 1

    if args.city_name != "None":
        args.data_root_path = os.path.join(datasets_path['NTHU']['data_root_path'], args.city_name)
        args.list_path = os.path.join(datasets_path['NTHU']['list_path'], args.city_name, 'List')
        args.target_crop_size = (1024,512)
        args.target_base_size = (1024,512)
    args.crop_size = args.target_crop_size
    args.base_size = args.target_base_size

    agent = Evaluater(cuda=True, train_id="train_id", logger=logger, **vars(args))
    agent.main()