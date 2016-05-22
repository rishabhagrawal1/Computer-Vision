%%reading image file and creating SIFT descriptors%%
I = im2double(rgb2gray(imread('F:\vision\assign_2\dog.jpg')));
[ pos, scale, orient, desc ] = SIFT(I,4,2,ones(size(I)),0.02,10.0,2);
[database] = add_descriptors_to_database( I, pos, orient, scale, desc);

%%rotating about origin by 45 degrees%%
t =    [ 0.7072    0.7072         0
        -0.7072    0.7072         0
         0            0           1]
%%reflect about y axis%%
 t1 =   [ -1       0         0
           0       1         0
           0         0         1]
     
%%current demo using rotate about origin by 45 degrees%%       
[affineI1,mI]= imWarpAffine(I,t,0);  
%to remove the NaN added by the matlab in non used places in the image%
affineI1(isnan(affineI1)) = 0;
showIm(affineI1)

%%Extracting Sift features for second%%
[ pos1, scale1, orient1, desc1 ] = SIFT(affineI1,4,2,ones(size(affineI1)),0.02,10.0,2);

%% Hough Transform %%
[im_idx trans rot rho idesc inn wght] = hough( database, pos1, orient1, scale1, desc1);

%% Feeding position and scale to fit affine function %%
[max1, index] = max(wght);
desc_test = idesc{index}; 
desc_original = inn{index};
[w, h] = size(desc_test);
p0 = zeros(2,w);
p1 = zeros(2,w);
w_scale = zeros(1,w);
for(i = 1:w)
    p0(1,i) = database(im_idx(1)).pos(desc_original(i),1);
    p0(2,i) = database(im_idx(1)).pos(desc_original(i),2);
    w_scale(1,i) = database(im_idx(1)).scale(desc_original(i),1);
    p1(1,i) = pos1(desc_test(i),1);
    p1(2,i) = pos1(desc_test(i),2);
end

[aff] = fit_robust_affine_transform(p0, p1, w_scale,0.5)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Now using the affine matrix received to rotate the%%%
%%%%%%%%%%%%%%%% original image %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
t5 =   [    0.7073   -0.7126  133.0012
   0.7070    0.7084   -0.1274
      0             0          1.0000
]

[affineBack,mI1]= imWarpAffine(affineI1,t5,0);
%to remove the NaN added by the matlab in non used places in the image%
affineBack(isnan(affineBack)) = 0;
imshow(affineBack)
%}