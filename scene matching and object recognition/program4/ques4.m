%% scores1 are scores for all 49 sliding windows with test image of set1 %%
%% scores2 are scores for all 49 sliding windows with test image of set1 %%
%% index1 is the index for matching sliding window in test image of set1 %%
%% index2 is the index for matching sliding window in test image of set2 %%

function [scores1 scores2 index1 index2] = ObjectRecognition()
    %% Original IMAGE Parameters %%
    xOrig = 720;
    yOrig = 1280;
    %% Resized IMAGE Parameters %%
    xRes = 180;
    yRes = 320;
        
    [I] = loadImages(xRes, yRes);
    
    hogForSVM = []; 
    %% generate Hog feature map and save feature map images %%
    for(i = 1:size(I))
        %%Convert image into single precision class%%
        Im = I{i};
        ImSingle = im2single( Im );
        
        %%Apply HOG feature extractor%%
        cellSize = 8;
        [hog] = vl_hog(ImSingle, cellSize); %'verbose');
        
        %% saving the output HOG feature image %%
        Imhog = vl_hog('render', hog); % 'verbose');
        fileName = sprintf('%d.bmp',i);
        imwrite(Imhog, fileName, 'bmp');
        %% showing the HOG feature Image %%
        %clf ;imshow(Imhog); colormap gray;
        
        %% Transformm HOG mtrix to vector and then to matrix for all images %%
        [x,y,z] = size(hog);
        reshapeHog = reshape(hog, x*y*z, 1);
        [hogForSVM] = [hogForSVM reshapeHog];
    end
    %% Inputting this hog matrix for all images to SVM to train SVM %%
    labels = []; %% vector representing the objects in HOG feature matrix %%
    for (j = 1:size(I))
        k = -1;
        if(j <= 10)
            k = 1;
        end
        labels = [labels k];
    end
    
    [w b] = vl_svmtrain(hogForSVM, labels, 0.001, 'MaxNumIterations', 10000);
    
    %% Read test files%%
    Test1 = im2double(rgb2gray(imread('D:\vision\ques4\set1\test1.jpg')));
    Test2 = im2double(rgb2gray(imread('D:\vision\ques4\set2\test2.jpg')));
    Test = {im2single(Test1), im2single(Test2)};

    %% Use SVM to categorize test images with sliding window and save scores for all %%
    Itest = Test{1};
    scores1 = [];
    for(i = 1: ((xOrig/xRes)*2 -1))
        for(j = 1: ((yOrig/yRes)*2 -1))
            Islide = Itest((yRes*(j-1)/2)+1:yRes*(j+1)/2,(xRes*(i-1)/2)+1:xRes*(i+1)/2);
            hogTest = vl_hog(Islide, cellSize);
            [x,y,z] = size(hogTest);
            reshapedHogTest = reshape(hogTest, x*y*z, 1);
            scores1 = [scores1 (w'*reshapedHogTest + b)];
        end
    end
    
    Itest = Test{2};
    scores2 = [];
    for(i = 1: ((xOrig/xRes)*2 -1))
        for(j = 1: ((yOrig/yRes)*2 -1))
            Islide = Itest((yRes*(j-1)/2)+1:yRes*(j+1)/2,(xRes*(i-1)/2)+1:xRes*(i+1)/2);
            hogTest = vl_hog(Islide, cellSize);
            [x,y,z] = size(hogTest);
            reshapedHogTest = reshape(hogTest, x*y*z, 1);
            scores2 = [scores2 (w'*reshapedHogTest + b)];
        end
    end
    %% Matching window in %%
    [max1 index1] = max(abs(scores1(:)));
    [max2 index2] = max(abs(scores2(:)));
end

function [I] = loadImages(x, y)
    %%Model1%%
    I1 = im2double(rgb2gray(imread('D:\vision\ques4\set1\1.jpg')));
    I1 = imresize(I1,[x,y]);
    I2 = im2double(rgb2gray(imread('D:\vision\ques4\set1\2.jpg')));
    I2 = imresize(I2,[x,y]);
    I3 = im2double(rgb2gray(imread('D:\vision\ques4\set1\3.jpg')));
    I3 = imresize(I3,[x,y]);
    I4 = im2double(rgb2gray(imread('D:\vision\ques4\set1\4.jpg')));
    I4 = imresize(I4,[x,y]);
    I5 = im2double(rgb2gray(imread('D:\vision\ques4\set1\5.jpg')));
    I5 = imresize(I5,[x,y]);
    I6 = im2double(rgb2gray(imread('D:\vision\ques4\set1\6.jpg')));
    I6 = imresize(I6,[x,y]);
    I7 = im2double(rgb2gray(imread('D:\vision\ques4\set1\7.jpg')));
    I7 = imresize(I7,[x,y]);
    I8 = im2double(rgb2gray(imread('D:\vision\ques4\set1\8.jpg')));
    I8 = imresize(I8,[x,y]);
    I9 = im2double(rgb2gray(imread('D:\vision\ques4\set1\9.jpg')));
    I9 = imresize(I9,[x,y]);
    I10 = im2double(rgb2gray(imread('D:\vision\ques4\set1\10.jpg')));
    I10 = imresize(I10,[x,y]);
    
    %%Model 2%%
    J1 = im2double(rgb2gray(imread('D:\vision\ques4\set2\11.jpg')));
    J1 = imresize(J1,[x,y]);
    J2 = im2double(rgb2gray(imread('D:\vision\ques4\set2\12.jpg')));
    J2 = imresize(J2,[x,y]);
    J3 = im2double(rgb2gray(imread('D:\vision\ques4\set2\13.jpg')));
    J3 = imresize(J3,[x,y]);
    J4 = im2double(rgb2gray(imread('D:\vision\ques4\set2\14.jpg')));
    J4 = imresize(J4,[x,y]);
    J5 = im2double(rgb2gray(imread('D:\vision\ques4\set2\15.jpg')));
    J5 = imresize(J5,[x,y]);
    J6 = im2double(rgb2gray(imread('D:\vision\ques4\set2\16.jpg')));
    J6 = imresize(J6,[x,y]);
    J7 = im2double(rgb2gray(imread('D:\vision\ques4\set2\17.jpg')));
    J7 = imresize(J7,[x,y]);
    J8 = im2double(rgb2gray(imread('D:\vision\ques4\set2\18.jpg')));
    J8 = imresize(J8,[x,y]);
    J9 = im2double(rgb2gray(imread('D:\vision\ques4\set2\19.jpg')));
    J9 = imresize(J9,[x,y]);
    J10 = im2double(rgb2gray(imread('D:\vision\ques4\set2\20.jpg')));
    J10 = imresize(J10,[x,y]);
    I = {I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,J1,J2,J3,J4,J5,J6,J7,J8,J9,J10};
end