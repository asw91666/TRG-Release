from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', default=True, help='use eval mode during test time.')
        parser.add_argument('--seg', action='store_true',  help='use seg_pred during test time.')
        #parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        # loss option
        parser.add_argument('--rotmat_loss_weight', type=float, default=10.0, help='')
        parser.add_argument('--lmk5_2d_loss_weight', type=float, default=0.01, help='')
        parser.add_argument('--loop_loss_weight', type=float, default=0.5, help='')
        parser.add_argument('--tta_iter', type=int, default=50, help='')
        parser.add_argument('--tta_threshold', type=float, default=3.0, help='')
        parser.add_argument('--tta_output_dir', type=str, default='./dataset/300W_LP/tta_pgt', help='')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
