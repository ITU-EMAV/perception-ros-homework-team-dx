import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv_1x1 = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride=1)

        # Encoder side
        # First layer two convolution layers
        self.conv_down1 = self.double_conv(3,32,3)
        # Second layer two convolution layers
        self.conv_down2 = self.double_conv(32,64,3)
        # Third layer two convolution layers
        self.conv_down3 = self.double_conv(64,128,3)
        # Fourth layer two convolution layers
        self.conv_down4 = self.double_conv(128,256,3)
        # Fifth layer two convlution layers
        self.conv_down5 = self.double_conv(256,512,3)
        # Sixth layer two convolution layers
        self.conv_down6 = self.double_conv(512,1024,3)

        # Decoder Side
        # First decoder layer two convolution layers
        self.upsample1 = nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
        self.conv_up1 = self.double_conv(1024,512,3)
        # Second decoder layer two convolution layers
        self.upsample2 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
        self.conv_up2 = self.double_conv(512,256,3)
        # Third decoder layer two convolution layers
        self.upsample3 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.conv_up3 = self.double_conv(256,128,3)
        # Fourth decoder layer two convolution layers
        self.upsample4 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        self.conv_up4 = self.double_conv(128,64,3)
        # Fifth decoder layer two convolution layers
        self.upsample5 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2)
        self.conv_up5 = self.double_conv(64,32,3)
        # Sixth decoder layer two convolution layers
        self.conv_up6 = self.double_conv(32,3,3)


    def forward(self,x): # Use F.interpolate on the final output to resize back to the original size
        # Encoder
        x1 = self.conv_down1(x)     # Output from first double conv
        x1_pool = self.max_pool(x1)
        x2 = self.conv_down2(x1_pool)     # Output from second double conv
        x2_pool = self.max_pool(x2)
        x3 = self.conv_down3(x2_pool)     # Output from third double conv
        x3_pool = self.max_pool(x3)
        x4 = self.conv_down4(x3_pool)     # Output from fourth double conv
        x4_pool = self.max_pool(x4)
        x5 = self.conv_down5(x4_pool)     # Output from fifth double conv
        x5_pool = self.max_pool(x5)
        x6 = self.conv_down6(x5_pool)     # Output from sixth double conv

        # print(x6.shape)
        # Decoder
        # Decoder first layer
        x7 = self.upsample1(x6)
        (_,_,H,W) = x7.shape
        x5_cropped = self.crop(x5,(H,W))
        x8 = self.conv_up1(torch.cat((x5_cropped,x7),dim=1))
        # Decoder second layer
        x9 = self.upsample2(x8)
        (_,_,H,W) = x9.shape
        x4_cropped = self.crop(x4,(H,W))
        x10 = self.conv_up2(torch.cat((x4_cropped,x9),dim=1))
        # Decoder third layer
        x11 = self.upsample3(x10)
        (_,_,H,W) = x11.shape
        x3_cropped = self.crop(x3,(H,W))
        x12 = self.conv_up3(torch.cat((x3_cropped,x11),dim=1))
        # Decoder fourth layer
        x13 = self.upsample4(x12)
        (_,_,H,W) = x13.shape
        x2_cropped = self.crop(x2,(H,W))
        x14 = self.conv_up4(torch.cat((x2_cropped,x13),dim=1))
        # Decoder fifth layer
        x15 = self.upsample5(x14)
        (_,_,H,W) = x15.shape
        x1_cropped = self.crop(x1,(H,W))
        x16 = self.conv_up5(torch.cat((x1_cropped,x15),dim=1))

        # x15 = self.conv_1x1(x14)
        x17 = self.conv_1x1(x16)

        # x16 = F.interpolate(x15,(720,1280))
        x18 = F.interpolate(x17,(180,330))
        # return x16
        return x18

    def double_conv(self,in_c,out_c,k_size=3):
        conv_double = nn.Sequential(
            nn.Conv2d(in_c,out_c,k_size,1,1,bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,k_size,1,1,bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        return conv_double

    def crop(self,image,target_size):
        width = target_size[1]
        height = target_size[0]
        (_,_,height_img,width_img) = image.shape

        delta_width = ((width_img - width)//2)
        delta_height = height_img - height

        if (width_img - 2*delta_width) > width:
            cropped_image = image[:,:,delta_height:height_img,delta_width:width_img-delta_width-1]
        elif (width_img - 2*delta_width) < width:
            cropped_image = image[:,:,delta_height:height_img,delta_width-1:width_img-delta_width]
        else:
            cropped_image = image[:,:,delta_height:height_img,delta_width:width_img-delta_width]

        return cropped_image
    
class EnsembleUNet(nn.Module):
    def __init__(self, unet1, unet2):
        super(EnsembleUNet,self).__init__()

        self.conv_decision_1 = nn.ConvTranspose2d(in_channels=2,out_channels=12,kernel_size=3,stride=1,bias=True)
        self.conv_decision_2 = nn.Conv2d(in_channels=12,out_channels=6,kernel_size=3,stride=1,bias=True)
        self.conv_decision_3 = nn.Conv2d(in_channels=6,out_channels=1,kernel_size=3,stride=1,padding='same',bias=True)

        self.unet_1 = unet1
        self.unet_2 = unet2

    def forward(self,x):

        x1 = torch.clone(x)
        x2 = torch.clone(x)

        out1 = self.unet_1.forward(x1)
        out2 = self.unet_2.forward(x2)

        combined_out = torch.cat((out1,out2),dim=1)

        out = self.conv_decision_1(combined_out)
        out = F.relu(out,inplace=True)
        out = self.conv_decision_2(out)
        out = F.relu(out,inplace=True)
        out = self.conv_decision_3(out)
#         out = torch.sigmoid(out)

        return out