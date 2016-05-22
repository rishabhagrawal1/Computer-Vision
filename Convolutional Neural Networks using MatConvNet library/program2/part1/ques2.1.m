function ques_1_of2(varargin)

setup ;
run('D:\vision\vlfeat-0.9.20\toolbox\vl_setup')
addpath D:\vision\assign_3\cifar-10-batches-mat

% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load CIFAR Image set
CIFAR_DIR='D:\vision\assign_3\cifar-10-batches-mat';

%% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/data_batch_1.mat']);
f2=load([CIFAR_DIR '/data_batch_2.mat']);
f3=load([CIFAR_DIR '/data_batch_3.mat']);
f4=load([CIFAR_DIR '/data_batch_4.mat']);
f5=load([CIFAR_DIR '/data_batch_5.mat']);
f6=load([CIFAR_DIR '/test_batch.mat']); 
data = ([f1.data; f2.data; f3.data; f4.data; f5.data; f6.data]);
clear f1 f2 f3 f4 f5 f6;

imdb.images.label = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels;f6.labels]');

[nImags,nPixels] = size(data)

%%Loading the data
for(i = 1 : nImags)
    imt = data(i,:)
    imaget(:,:,1)=reshape(imt(1 : n/3),32,32);
    imaget(:,:,2)=reshape(imt(1+n/3 : n*2/3),32,32);
    imaget(:,:,3)=reshape(imt(1+n*2/3 : n),32,32);
    imdb.images.data(:,:,:,i) = im2single(imaget);
    imdb.images.id(1,i) = i;
    if(i<=nImags*5/6)
      imdb.images.set(1,i) = 1;      
    else
      imdb.images.set(1,i) = 2;
    end
end

%%Saving class information
imdb.meta.classes = '0123456789';

% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = modifiedInitializeCNN() ;

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

trainOpts.batchSize = 100 ;
trainOpts.numEpochs = 15 ;
trainOpts.continue = true ;
trainOpts.useGpu = false ;
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = 'experiment/cifar_experiment' ;
trainOpts = vl_argparse(trainOpts, varargin);

% Take the average image out
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;

% Convert to a GPU array if needed
if trainOpts.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Move the CNN back to the CPU if it was trained on the GPU
if trainOpts.useGpu
  net = vl_simplenn_move(net, 'cpu') ;
end

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save('experiment/cifar_experiment/cifarcnn.mat', '-struct', 'net') ;

% -------------------------------------------------------------------------
% Part 4.4: Load the model
% -------------------------------------------------------------------------

% Load the CNN learned before
net = load('experiment/cifar_experiment/cifarcnn.mat') ;

% -------------------------------------------------------------------------
% Part 4.5: visualize the learned filters
% -------------------------------------------------------------------------
figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.filters),'spacing',2)
axis equal ; title('filters in the first layer') ;

% -------------------------------------------------------------------------
% 4.6 use the model to classify an image
% -------------------------------------------------------------------------
% obtain and preprocess an image data
train = find(imdb.images.set == 1) ;
test = find(imdb.images.set == 2) ;
for (i = numel(train)+1: numel(test))
    im1 = imdb.images.data(:,:,:,1) ;
    im1 = 256 * (im1 - net.imageMean) ;
    % run the CNN
    res = vl_simplenn(net, im1) ;
end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
im = 256 * reshape(im, 32, 32, 3, []) ;
labels = imdb.images.label(1,batch) ;
