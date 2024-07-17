import torch
import torch.nn as nn
from .model_repository import Resnet18_8s
import math
import torch.nn.functional as F

class DeformNetUV2(nn.Module):
    def __init__(self, grid, nv_prior=1220, img_size=192):
        super(DeformNetUV2, self).__init__()
        self.cnn = Resnet18_8s()
        self.pos_encod=self.positionalencoding2d(64, 192, 192)
        self.grid = grid
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.instance_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.category_global = nn.Sequential(
            nn.Conv1d(32, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.assignment = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, nv_prior, 1),
        )

    def forward(self, img, choose):
        """
        Args:
            img: bs x 3 x H x W
            choose: bs x n_pts
            cat_id: bs
            prior: bs x nv x 3

        Returns:
            assign_mat: bs x n_pts x nv
            inst_shape: bs x nv x 3
            deltas: bs x nv x 3
            log_assign: bs x n_pts x nv, for numerical stability

        """
        #bs, n_pts = points.size()[:2]
        bs,n_pts =choose.size()

        nv = self.grid.shape[2]

        out_img, seg_pred, uv_feat, uv_pred  = self.cnn(img)
        di = out_img.size()[1]
        emb = out_img.view(bs, di, -1)
        
        # print(seg_pred.size(),out_img.size())
        choose = choose.unsqueeze(1).repeat(1, di*2, 1)
        # print(choose.size())
        emb = torch.gather(emb, 2, choose[:,0:di,:]).contiguous()
        # print(emb.size())
        emb = self.instance_color(emb)   #bs*64*n_pts
        pos_encod=self.pos_encod.unsqueeze(0).repeat(bs,1,1,1)
        #dj = pos_encod.size()[1]
        pos_encod=pos_encod.view(bs,di*2,-1)
        pos_encod = torch.gather(pos_encod, 2, choose).contiguous() #bs*64*n_pts
        inst_local = torch.cat((emb, pos_encod), dim=1)     # bs x 128 x n_pts
        inst_global = self.instance_global(inst_local)    # bs x 1024 x 1
        uv_local = F.grid_sample(uv_feat, self.grid.expand(bs, -1, -1, -1), align_corners=False)   
        uv_local = uv_local.squeeze(2)


        cat_global = self.category_global(uv_local)  # bs x 1024 x 1
        # print('cat_global',cat_global.size())

        # assignemnt matrix
        assign_feat = torch.cat((inst_local, inst_global.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
        assign_mat = self.assignment(assign_feat)
        assign_mat = assign_mat.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        #index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        #assign_mat = torch.index_select(assign_mat, 0, index)   # bs x nv x n_pts
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()    # bs x n_pts x nv

        return assign_mat,  seg_pred, uv_pred

    def positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                    "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width).to(device='cuda')
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe


