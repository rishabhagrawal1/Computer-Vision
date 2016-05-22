%%image 1%%
I1 = im2double(rgb2gray(imread('D:\vision\assign_2\ques2\question2_non_rotated.jpg')));
[ pos, scale, orient, desc ] = SIFT(I1,4,2,ones(size(I1)),0.02,10.0,2);
[database] = add_descriptors_to_database( I1, pos, orient, scale, desc);

%%image 2%% 
I2 = im2double(rgb2gray(imread('D:\vision\assign_2\ques2\question2_rotated.jpg')));
[ pos1, scale1, orient1, desc1 ] = SIFT(I2,4,2,ones(size(I2)),0.02,10.0,1);

%% hough Transform %%
[im_idx trans rot rho idesc inn wght] = hough( database, pos1, orient1, scale1, desc1);
[max1, index] = max(wght);
desc_test = idesc{index}; 
desc_original = inn{index};

%%creating pos and weight matrix%%
[w, h] = size(desc_test);
p0 = zeros(2,w);
p1 = zeros(2,w);
w_scale = zeros(1,w);
for(i = 1:w)
    p0(1,i) = database.pos(desc_original(i),1);
    p0(2,i) = database.pos(desc_original(i),2);
    w_scale(1,i) = database.scale(desc_original(i),1);
    p1(1,i) = pos1(desc_test(i),1);
    p1(2,i) = pos1(desc_test(i),2);
end

%% fit affine transform %%
[aff] = fit_robust_affine_transform(p0, p1, w_scale,0.5)
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%Checking received affine matrix%%%%%%%%%%%%%%%%%%
%{

%% Affine transformation received by rotating about origin by 45 degrees%%
t = [    0.6931   -0.5491  118.3307
         0.5328    0.6889   -5.7430
            0         0    1.0000]

%% Applying on original Image %%
[affineI1,mI]= imWarpAffine(I1,t,0);
%to remove the NaN added by the matlab in non used places in the image%
%affineI1(isnan(affineI1)) = 0;

%% Applying on other Image%%
[affineI2,mI]= imWarpAffine(I2,t,0);
%to remove the NaN added by the matlab in non used places in the image%
%affineI2(isnan(affineI2)) = 0;

%pairOfImages = [affineI1, affineI2]; % or [I1;I2] 
imshow(pairOfImages);
%}