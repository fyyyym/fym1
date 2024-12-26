# @brief:     Generic pytorch module for NN
# @author     Kaustab Pal    [kaustab21@gmail.com]

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time
from atppnet.models.base import BasePredictionModel
from atppnet.models.blocks import ConvLSTM, LinearAttentionBlock,ResContextBlock
from atppnet.models.blocks import DownBlock, UpBlock, CustomConv2d, CNN3D_block,MultiModalAttentionFusion
import random
import numpy as np

class ATPPNet(BasePredictionModel):
    def __init__(self, cfg, num_channels, num_kernels, kernel_size, padding, 
    activation, img_size, num_layers, peep=True):
        super(ATPPNet, self).__init__(cfg)
        self.channels = self.cfg["MODEL"]["CHANNELS"]
        self.skip_if_channel_size = self.cfg["MODEL"]["SKIP_IF_CHANNEL_SIZE"]
        self.num_kernels = self.channels[-1]
        self.img_size = img_size
        self.batch = self.cfg["TRAIN"]["BATCH_SIZE"] 
        self.height = self.cfg['DATA_CONFIG']['HEIGHT']
        self.width = self.cfg['DATA_CONFIG']['WIDTH']
        self.n_past_steps=self.cfg["MODEL"]["N_PAST_STEPS"]
        self.n_future_steps=self.cfg["MODEL"]["N_FUTURE_STEPS"]
        self.res_num = self.cfg["MODEL"]["RES_NUM"]
        frame_h = self.img_size[0]//2**(len(self.channels)-1)
        frame_w = self.img_size[1]//2**(len(self.channels)-1)
        self.frame_size = (frame_h, frame_w)
        self.CNN3D_block = CNN3D_block(cfg = self.cfg)

        self.conv_skip = CustomConv2d(
            2 * self.channels[-1],
            self.channels[-1],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
            circular_padding=True,
        )
        '''
        self.downCntx = ResContextBlock(self.n_past_steps, self.channels[0])
        self.downCntx2 = ResContextBlock(self.channels[0], self.channels[0])
        self.downCntx3 = ResContextBlock(self.channels[0], self.channels[0])

        self.metaConv = MetaKernel(num_batch=int(self.batch/ torch.cuda.device_count()) ,
                                   feat_height=self.height ,
                                   feat_width=self.width,
                                   coord_channels=self.n_past_steps)
        '''
        self.RI_downCntx = ResContextBlock(1, self.channels[0])
        self.RI_DownLayers = nn.ModuleList()
        for i in range(len(self.channels)-1):
            if self.channels[i + 1] in self.skip_if_channel_size:    
                self.RI_DownLayers.append(
                        DownBlock(
                            cfg = self.cfg,
                            in_channels = self.channels[i],
                            out_channels = self.channels[i + 1],
                            skip=True,
                        )
                    )
            else :
                self.RI_DownLayers.append(
                        DownBlock(
                            cfg = self.cfg,
                            in_channels = self.channels[i],
                            out_channels = self.channels[i + 1],
                            skip=False
                        )
                    )  

                                    

        self.input_layer = CustomConv2d(
            num_channels,
            self.channels[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
            circular_padding=True,
        )

        self.DownLayers = nn.ModuleList()
        self.UpLayers = nn.ModuleList()


        for i in range(len(self.channels) - 1):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.DownLayers.append(
                        DownBlock(
                            cfg = self.cfg,
                            in_channels = self.channels[i],
                            out_channels = self.channels[i + 1],
                            skip=True,
                        )
                    )
            else:
                self.DownLayers.append(
                        DownBlock(
                            cfg = self.cfg,
                            in_channels = self.channels[i],
                            out_channels = self.channels[i + 1],
                            skip=False,
                        )
                    )

        for i in reversed(range(len(self.channels) - 1)):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.UpLayers.append(
                    UpBlock(
                        cfg = self.cfg,
                        in_channels = self.channels[i + 1],
                        out_channels = self.channels[i],
                        skip=True,
                    )
                )
            else:
                self.UpLayers.append(
                    UpBlock(
                        cfg = self.cfg,
                        in_channels = self.channels[i + 1],
                        out_channels = self.channels[i],
                        skip=False,
                    )
                )
        self.rv_head = CustomConv2d(
            self.channels[0],
            2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
            circular_padding=True,
        )

        self.mos_head = CustomConv2d(
            self.channels[0],
            2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
            circular_padding=True,
        )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.motion_encoder = nn.ModuleList()
        self.motion_decoder = nn.ModuleList()
        for i in range(len(self.channels)):
            if self.channels[i] in self.skip_if_channel_size:
                self.encoder.append(
                        ConvLSTM(
                            input_dim=self.channels[i],
                            hidden_dim=self.channels[i],
                            kernel_size=kernel_size,
                            padding=padding, activation=activation,
                            frame_size=self.frame_size, num_layers=num_layers,
                            peep=peep, return_all_layers=True)
                        )
                self.motion_encoder.append(
                        ConvLSTM(
                            input_dim=self.channels[i],
                            hidden_dim=self.channels[i],
                            kernel_size=kernel_size,
                            padding=padding, activation=activation,
                            frame_size=self.frame_size, num_layers=num_layers,
                            peep=peep, return_all_layers=True)
                        )
                self.attention.append(
                         MultiModalAttentionFusion(feature_dim=self.channels[i]))
                        
                self.decoder.append(
                        ConvLSTM(
                            input_dim=self.channels[i],
                            hidden_dim=self.channels[i],
                            kernel_size=kernel_size,
                            padding=padding, activation=activation,
                            frame_size=self.frame_size, num_layers=num_layers,
                            peep=peep, return_all_layers=True)
                        )
                self.motion_decoder.append(
                        ConvLSTM(
                            input_dim=self.channels[i],
                            hidden_dim=self.channels[i],
                            kernel_size=kernel_size,
                            padding=padding, activation=activation,
                            frame_size=self.frame_size, num_layers=num_layers,
                            peep=peep, return_all_layers=True)
                        )
        self.encoder.append(
                ConvLSTM(
                    input_dim=self.channels[-1],
                    hidden_dim=self.channels[-1],
                    kernel_size=kernel_size,
                    padding=padding, activation=activation,
                    frame_size=self.frame_size, num_layers=num_layers,
                    peep=peep, return_all_layers=True)
                )
        self.motion_encoder.append(
                    ConvLSTM(
                        input_dim=self.channels[i],
                        hidden_dim=self.channels[i],
                        kernel_size=kernel_size,
                        padding=padding, activation=activation,
                        frame_size=self.frame_size, num_layers=num_layers,
                        peep=peep, return_all_layers=True)
                        )
        self.attention.append(
                MultiModalAttentionFusion(feature_dim=self.channels[-1])
        )
        self.decoder.append(
                ConvLSTM(
                    input_dim=self.channels[-1],
                    hidden_dim=self.channels[-1],
                    kernel_size=kernel_size,
                    padding=padding, activation=activation,
                    frame_size=self.frame_size, num_layers=num_layers,
                    peep=peep, return_all_layers=True)
                )
        self.motion_decoder.append(
                    ConvLSTM(
                        input_dim=self.channels[i],
                        hidden_dim=self.channels[i],
                        kernel_size=kernel_size,
                        padding=padding, activation=activation,
                        frame_size=self.frame_size, num_layers=num_layers,
                        peep=peep, return_all_layers=True)
                     )

        self.norm = nn.BatchNorm3d(num_features=self.cfg["MODEL"]["N_PAST_STEPS"])



    def forward(self, x, x_res):
        x = x[:, :, self.inputs, :, :]
        device = x.device
        batch, seq_length, num_channels, height, width = x.shape
        # if you want to normalize the input uncomment the below lines
        #past_mask = x != -1.0

        ## Standardization and set invalid points to zero
        #mean = self.mean[None, self.inputs, None, None, None]

        #std = self.std[None, self.inputs, None, None, None]
        #x = torch.true_divide(x - mean, std)
        #x = x * past_mask
        #SOS = torch.zeros((batch,num_channels, height, width), device=device)
        x_res.unsqueeze(1)
        x_res = x_res.view(batch*seq_length, 1, height, width).to(device)
        x_res = self.RI_downCntx(x_res)
        res_layers=[]
        for l in self.RI_DownLayers:
            x_res = l(x_res)
            if x_res.shape[1] in self.skip_if_channel_size :                
                res_layers.append(x_res)
        res_layers.append(x_res)

        skip_layer_encoder = []
        prob_skip_layer_encoder = []
        skip_layer_decoder = []
        attn_list = []

        x = x.view(batch*seq_length, num_channels, height, width).to(device)
        x = self.input_layer(x)
        for l in self.DownLayers:
            x = l(x)
            if l.skip:
                skip_layer_encoder.append(x)
        skip_layer_encoder.append(x) #添加了 32 64 128channels的x
        _,c,h,w = x.shape
        
        tc_x = x.view(batch,seq_length,c,h,w)#怎么操作？？
        x_res = x_res.view(batch,seq_length,c,h,w)
        x_mf = torch.cat((tc_x,x_res),dim=2)
        x_mf = self.CNN3D_block(x_mf)
        for s in range(len(skip_layer_encoder)):
            _,c,h,w = skip_layer_encoder[s].shape

            #-------LSTM motion -----
            res_layers[s] =\
                    res_layers[s].view(batch,seq_length,c,h,w).to(device)
            motion_stack, mos_h = self.motion_encoder[s](res_layers[s])
            motion = motion_stack[-1] 
            motion = self.norm(motion) #最后一层输出的结果（h1-5）
            last_motion = motion[:,-1] # final layer's hidden state


            #-------LSTM appearance -----
            skip_layer_encoder[s] =\
                    skip_layer_encoder[s].view(batch,seq_length,c,h,w).to(device)
            app_stack, app_h = self.encoder[s](skip_layer_encoder[s])
            app = app_stack[-1] 
            app = self.norm(app) #最后一层输出的结果（h1-5）
            last_app = app[:,-1] # final layer's hidden state
            
            app_out = torch.zeros((batch, self.cfg["MODEL"]["N_FUTURE_STEPS"],
                    last_app.shape[1], last_app.shape[2], last_app.shape[3]),device=device)
            
            motion_out = torch.zeros((batch, self.cfg["MODEL"]["N_FUTURE_STEPS"],
                    last_app.shape[1], last_app.shape[2], last_app.shape[3]),device=device)           

            #---对未来时间步预测
            for i in range(self.cfg["MODEL"]["N_FUTURE_STEPS"]):

                # ----利用运动信息的特征提取

                #fused_feature = self.attention[s](app,motion)  
                fused_feature = app*motion
            
                #----预测时间t时的特征块
                #           |------------------------------| 隐层不断迭代
                output, app_h = self.decoder[s](fused_feature, app_h)                                #  <-------------------------
                                                                                                #                           |
                                                                                               #                            |   用新的外观特征和运动特征预测下一个时间步
                                                                                               #                            |
                latest_t = output[-1][:,-1] #最后一层输出结果的最后一步，也就t=s时的预测结果                                          |  
                before_t = app[:,1:]         #利用解码器信息                                                 #                           |
                app = torch.cat((before_t,latest_t.unsqueeze(1)),1) #将外观特征输入h1-5-----》h2----    #   ------------------------


               #--------------motion predict
             # motion_stack, mos_h = self.motion_decoder[s](motion, mos_h)                                #  <-------------------------
                                                                                                #                           |
                                                                                               #                            |   预测下一个时间步的运动特征
                                                                                               #                            |
                lastest_motion = motion_stack[-1][:,-1] #最后一层输出结果的最后一步，也就t=s时的预测结果                                          |  
                motion = motion[:,1:]                                                          #                           |
                motion = torch.cat((motion,lastest_motion.unsqueeze(1)),1) #将外观特征输入h1-5-----》h2----    #   ------------------------


                #将t=s时候的结果存入
                app_out[:,i] = latest_t 
                motion_out[:,i] = lastest_motion 

            skip_layer_decoder.append(
                    app_out.view(
                        batch*self.cfg["MODEL"]["N_FUTURE_STEPS"],
                        app_out.shape[2],app_out.shape[3],app_out.shape[4]).to(device)
                    )
            

        x = skip_layer_decoder.pop()
        x = torch.cat((x, x_mf), dim=1)
        x = self.conv_skip(x)
        for l in self.UpLayers:
            if l.skip:
                x = l(x, skip_layer_decoder.pop())
            else:
                x = l(x)
        #--head      
        rv = self.rv_head(x)
        mos = self.mos_head(x)

        rv = rv.view(batch, self.cfg["MODEL"]["N_FUTURE_STEPS"], rv.shape[1], rv.shape[2], rv.shape[3]).to(device)
        mos = mos.view(batch, self.cfg["MODEL"]["N_FUTURE_STEPS"], mos.shape[1], mos.shape[2], mos.shape[3]).to(device)

        assert not torch.any(torch.isnan(x))
        output = {}
        output["rv"] = self.min_range + nn.Sigmoid()(rv[:, :, 0, :, :]) * (
            self.max_range - self.min_range
        )
        output["mask_logits"] = rv[:, :, 1, :, :]
        output["motion_seg"] = mos
        

        return output, attn_list

if __name__ == "__main__":
    config_filename = 'config/parameters.yml'# location of config file
    cfg = yaml.safe_load(open(config_filename))
    model = ATPPNet(cfg, num_channels=1, num_kernels=32, 
                    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
                    img_size=(64, 64), num_layers=3, peep=False)
    model = model.to("cuda")
    inp_nuscenes = torch.randn(2,5,1,32,1024)
    inp_kitti = torch.randn(2,5,1,64,2048)
    inp = inp_nuscenes.to("cuda")
    inf_time = []
    for i in range(52):
        start = time.time()
        pred, _ = model(inp)
        inf_time.append((time.time()-start)/inp.shape[0])
    inf_time = inf_time[2:]
    inf_time - np.array(inf_time)
    print("Inference time (sec): ", np.mean(inf_time))










