function ques_2_of2(varargin)

%%Addpath and run the setup scripts
addpath D:\vision\assign_3\libsvm-3.20\matlab
addpath D:\vision\assign_3\cifar-10-batches-mat
addpath D:\vision\vlfeat-0.9.20\toolbox\vl_setup
run ('D:\vision\assign_3\libsvm-3.20\matlab\make')
run('D:\vision\vlfeat-0.9.20\toolbox\vl_setup')

setup ;

% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load CIFAR Image set
CIFAR_DIR='D:\vision\assign_3\cifar-10-batches-mat';

%% Load CIFAR training and test data
%fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/data_batch_1.mat']);
f2=load([CIFAR_DIR '/data_batch_2.mat']);
f3=load([CIFAR_DIR '/data_batch_3.mat']);
f4=load([CIFAR_DIR '/data_batch_4.mat']);
f5=load([CIFAR_DIR '/data_batch_5.mat']);
f6=load([CIFAR_DIR '/test_batch.mat']); 
data = ([f1.data; f2.data; f3.data; f4.data; f5.data; f6.data]);

imdb.images.label = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels;f6.labels]');
clear f1 f2 f3 f4 f5 f6;

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
trainOpts = vl_argparse(trainOpts, {});

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
% Part 4.5: apply the model
% -------------------------------------------------------------------------
% Load the CNN learned before
net = load('experiment/cifar_experiment/cifarcnn.mat') ;

% -------------------------------------------------------------------------
% Training the SVM model
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
test = find(imdb.images.set == 2) ;

%%collect the training images feature descriptors
training_instance_matrix = []
for (i = 1: numel(train))
	%% Create Label Vector
    label = imdb.images.label(1,train(1,i)) ;
    training_label_vector(i,1) = label;
    
	%% Create Feature Vectors
	im = imdb.images.data(:,:,:,train(1,i)) ;
    im = 256 * (im - net.imageMean) ;
    
    % run the CNN  
    resTrain = vl_simplenn(net, im) ;
	featureV = resTrain(9).x;
    [x y nFeatureV] = size(featureV);
    featureV = reshape(featureV, x*y*nFeatureV, 1);
    training_instance_matrix = [training_instance_matrix featureV];  %%one way of creating vector using transpode below
end

%%training_instance_matrix
model = svmtrain(double(training_label_vector), double(training_instance_matrix') );

% -------------------------------------------------------------------------
% use the model to classify an image
% -------------------------------------------------------------------------
% obtain and preprocess an image data

for (i = 1 : numel(test))
    %% Create Label Vector
    label = imdb.images.label(1,test(1,i)) ;
    testing_label_vector(i,1) = label;

	%% Create Feature Vectors
    im = imdb.images.data(:,:,:,test(1,i));
    im = 256 * (im - imageMean) ;
	
	% run the CNN
    resTest = vl_simplenn(net, im) ;
    featureV = resTest(9).x;
    [x y nFeatureV] = size(featureV);
    testing_instance_matrix(i,:) = (reshape(featureV, x*y*nFeatureV ,1))';   %%other way of creating vector using transpode here only
end

%%Prediction Now
[predicted_label_test, accuracy_test, decision_value_test] = svmpredict(double(testing_label_vector), double(testing_instance_matrix), model)

[predicted_label_train, accuracy_train, decision_value_train] = svmpredict(double(testing_label_vector), double(testing_instance_matrix'), model)

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
im = 256 * reshape(im, 32, 32, 3, []) ;
labels = imdb.images.label(1,batch) ;
end
