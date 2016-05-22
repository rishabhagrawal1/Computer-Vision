%%Model 1%%
I1 = im2double(rgb2gray(imread('D:\vision\assign_2\ques3\model1i\obj1.jpg')));
[ pos, scale, orient, desc ] = SIFT(I1,4,2,ones(size(I1)),0.02,10.0,1);
[database] = add_descriptors_to_database( I1, pos, orient, scale, desc);

%%Model 2%%
J1 = im2double(rgb2gray(imread('D:\vision\assign_2\ques3\model2i\obj1.jpg')));
[ pos1, scale1, orient1, desc1 ] = SIFT(J1,4,2,ones(size(J1)),0.02,10.0,1);
[database] = add_descriptors_to_database( J1, pos1, orient1, scale1, desc1,database);

%%Loading other objects%%
I2 = im2double(rgb2gray(imread('D:\vision\assign_2\ques3\model1i\obj2.jpg')));
I2 = imresize(I2,[253,312]);
I3 = im2double(rgb2gray(imread('D:\vision\assign_2\ques3\model1i\obj3.jpg')));
I3 = imresize(I3,[253,312]);
I4 = im2double(rgb2gray(imread('D:\vision\assign_2\ques3\model1i\obj4.jpg')));
I4 = imresize(I4,[253,312]);
I5 = im2double(rgb2gray(imread('D:\vision\assign_2\ques3\model1i\obj5.jpg')));
I5 = imresize(I5,[253,312]);

%%Loading other objects%%
J2 = im2double(rgb2gray(imread('D:\vision\assign_2\ques3\model2i\obj2.jpg')));
J2 = imresize(J2,[253,312]);
J3 = im2double(rgb2gray(imread('D:\vision\assign_2\ques3\model2i\obj3.jpg')));
J3 = imresize(J3,[253,312]);
J4 = im2double(rgb2gray(imread('D:\vision\assign_2\ques3\model2i\obj4.jpg')));
J4 = imresize(J4,[253,312]);
J5 = im2double(rgb2gray(imread('D:\vision\assign_2\ques3\model2i\obj5.jpg')));
J5 = imresize(J5,[253,312]);

I = cat(5,I2,I3,I4,I5)
J = cat(5,J2,J3,J4,J5)

fid=fopen('MyFile.txt','w');

%% hough Transform %%
for(k = 1:4)
    %%SIFT%%
    temp = I(:,:,1,1,k);
    [ pos2, scale2, orient2, desc2 ] = SIFT(temp,4,2,ones(size(temp)),0.02,10.0,1);
    %%Hough%%
    [im_idx trans rot rho idesc inn wght] = hough( database, pos2, orient2, scale2, desc2);
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
        p1(1,i) = pos2(desc_test(i),1);
        p1(2,i) = pos2(desc_test(i),2);
    end

    %% fit affine transform %%
    [aff] = fit_robust_affine_transform(p0, p1, w_scale,0.5)
    
    fprintf(fid,'  %6.6f %12.6f %18.6f\n',aff);
    fprintf(fid, 'matched index in database is %d \n', im_idx);
end

for(k = 1:4)
    %%SIFT%%
    temp = J(:,:,1,1,k);
    [ pos2, scale2, orient2, desc2 ] = SIFT(temp,4,2,ones(size(temp)),0.02,10.0,1);
    %%Hough%%
    [im_idx trans rot rho idesc inn wght] = hough( database, pos2, orient2, scale2, desc2);
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
        p1(1,i) = pos2(desc_test(i),1);
        p1(2,i) = pos2(desc_test(i),2);
    end

    %% fit affine transform %%
    [aff] = fit_robust_affine_transform(p0, p1, w_scale,0.5)
    
    fprintf(fid,'  %6.6f %12.6f %18.6f\n',aff);
    fprintf(fid, 'matched index in database is %d \n', im_idx);
end

fclose(fid);
