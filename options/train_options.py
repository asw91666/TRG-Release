from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=8192, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display. Default is random window id')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=8192, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=8192, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', default=False, help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        #parser.add_argument('--evaluation_freq', type=int, default=5000, help='evaluation freq')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--seg', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--pretrained_name', type=str, default=None, help='resume training from another checkpoint')

        # training parameters
        parser.add_argument('--model_cfg_path', type=str, default='models/pymaf_model/configs/pymaf_face_config.yaml')
        parser.add_argument('--pretrained_weight_path', type=str, default='dcface_pretrain/checkpoint-23')
        parser.add_argument('--fine_tune', action='store_true', default=False)
        parser.add_argument('--vertex_world_weight', type=float, default=10.0, help='number of epochs with the initial learning rate')
        parser.add_argument('--vertex_cam_weight', type=float, default=1.0, help='number of epochs with the initial learning rate')
        parser.add_argument('--vertex_img_weight', type=float, default=0.001, help='number of epochs with the initial learning rate')
        parser.add_argument('--loop_loss_weight', type=float, default=0.25, help='number of epochs with the initial learning rate')
        parser.add_argument('--heatmap_loss_weight', type=float, default=0.01, help='number of epochs with the initial learning rate')
        parser.add_argument('--mask_loss_weight', type=float, default=0.1, help='number of epochs with the initial learning rate')
        parser.add_argument('--rotmat_loss_weight', type=float, default=1.0, help='')
        parser.add_argument('--scale_loss_weight', type=float, default=1.0, help='')
        parser.add_argument('--trans_loss_weight', type=float, default=1.0, help='')
        parser.add_argument('--lmk68_loss_weight', type=float, default=1.25, help='')
        parser.add_argument('--lmk_2d_loss_weight', type=float, default=1.0, help='')
        parser.add_argument('--edge_weight', type=float, default=0.1, help='')
        parser.add_argument('--pretrain_kp_weight', type=float, default=0.01, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=10, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        #parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        #parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lr_drop', type=int, default=20, help='when drop the learning rate')
        parser.add_argument('--random_sample', default=False, action='store_true', help='random sample data')
        parser.add_argument('--uniform_sample', default=False, action='store_true', help='random sample data')
        parser.add_argument('--only_use_backbone_w', default=False, action='store_true', help='random sample data')

        # scratch
        parser.add_argument('--scratch', default=False, action='store_true', help='random sample data')

        # evaluate
        parser.add_argument('--do_eval', default=False, action='store_true', help='Evaluate when training')

        self.isTrain = True
        return parser
