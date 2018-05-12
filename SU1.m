cd('F:\大三课程\视觉物联网\视觉物联网作业二\');

clear;
SE1 = [0,1;0,1];
SE2 = [1,1;1,1];
BW1=imread('1.jpg');
figure(1),
subplot(2,2,1);imshow(BW1);title('原图像');
BW11=rgb2gray(BW1);
BW2=edge(BW11,'roberts',0.18,'both');
subplot(2,2,2);imshow(BW2);title('边缘化处理图像（roberts算子）');
BW3=imerode(BW2,SE1); 
subplot(2,2,3);imshow(BW3);title('腐蚀图像');
BW4=imdilate(BW3,SE2); 
subplot(2,2,4);imshow(BW4);title('膨胀图像');


clear;
img=imread('2.bmp'); 
f=rgb2gray(img);
figure(2),
subplot(1,2,1);
imshow(f); title('原图像（高斯模糊）');
f=double(f);  
f=fft2(f);  
f=fftshift(f);  
[m,n]=size(f);  %  
d0=20;  
m1=fix(m/2);  
n1=fix(n/2);  
for i=1:m  
    for j=1:n  
        d=sqrt((i-m1)^2+(j-n1)^2);  
        h(i,j)=exp(-d^2/2/d0^2);  
    end  
end  
g=f.*h;  
g=ifftshift(g);  
g=ifft2(g);  
g=mat2gray(real(g));  
subplot(1,2,2);  
imshow(g);title('去模糊图像（高斯平滑滤波）');
imwrite(img,'2.jpg');


img= imread('3.bmp');
I = rgb2gray(img);
K = medfilt2(I);%采用二维中值滤波函数medfilt2对受椒盐噪声干扰的图像滤波 
figure(3),
subplot(1,2,1);imshow(I);
title('原图像（椒盐噪声）');
subplot(1,2,2);imshow(K);
title('去模糊图像（中值滤波）');



clear
rgb=imread('4.bmp');
% 对RGB每个通道进行histeq处理
r=rgb(:,:,1);
g=rgb(:,:,2);
b=rgb(:,:,3);
R=histeq(r);
G=histeq(g);
B=histeq(b);
result_rgb=cat(3,R,G,B);
% 结果显示
figure(4),
subplot(1,2,1),imshow(rgb),title('原图像（雾）')
subplot(1,2,2),imshow(result_rgb),title('去模糊图像（histeq:直方图均衡化）：')


H=imread('5.bmp');%读取图像　　　
I=rgb2gray(H);%将彩色图像转换为灰度图像
figure(5),
subplot(2,2,1);
imshow(I);%显示图像　　
title('原图像（需图像增强）');
subplot(2,2,3);
imhist(I);%绘制图像的灰度直方图　　　
title('原图的灰度直方图');
axis('auto');
subplot(2,2,2);
J=histeq(I,64);%对图像进行均衡化处理，返回有64级灰度的图像J　　　
imshow(J);%显示图像　　　
title('去模糊图像（直方图均衡化）');
subplot(2,2,4);
imhist(J);%绘制图像的灰度直方图　　
title('均衡后的灰度直方图');


Image=imread('6.jpg');  
[M,N,nDims]=size(Image);
Image=im2double(Image);% 获取图像的尺寸和波段数
ImageStretch=Image;
for i=1:nDims  % 对每个波段依次进行灰度拉伸
    Sp=Image(:,:,i);
    MaxDN=max(max(Sp));
    MinDN=min(min(Sp));
    Sp=(Sp-MinDN)/(MaxDN-MinDN);  % 灰度拉伸公式
    ImageStretch(:,:,i)=Sp;
end
figure(6),
subplot(1,2,1),imshow(Image);title('原图像（需对比度拉伸）');
subplot(1,2,2),imshow(ImageStretch);title('去模糊图像（灰度拉伸）');


I = im2double(imread('7.tif'));
figure(7),
subplot(1,2,1),imshow(I);
title('原图像（运动模糊）');
LEN = 8;
THETA = 0;
PSF = fspecial('motion', LEN, THETA);
wnr1 = deconvwnr(I, PSF, 0);
subplot(1,2,2),imshow(wnr1);
title('去模糊图像（维纳滤波）')


