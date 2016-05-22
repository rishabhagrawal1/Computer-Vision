function imdb_data(varargin)

%%Addpath and run the setup scripts
    addpath D:\vision\assign_3\libsvm-3.20\matlab\
    croppedImagePath = 'D:\vision\assign_3\CroppedImages'
    category = dir(croppedImagePath);
    imCount = 1
    setup ;
 	
    %%Saving the images, laabels, sets etc for using later
    imdb.images.label=[]

    %%Have to remove the first two listing of the directory
    for i = 1: length(category)
        if category(i).name(1) == '.' 
            continue
        else
            imageFiles = dir(fullfile(croppedImagePath, category(i).name,'image*'));
            nImages = min(50, length(imageFile));
            for j = 1: nImages
                if imageFiles(j).name(1) == '.'
                    continue
                else
                    im = imread(fullfile(croppedImagePath,category(i).name, imageFiles(j).name));
                    imdb.images.data(:,:,:,imCount) = im2single(im);
                    imdb.images.id(1,imCount) = imCount;
                    if(j<=30)
                        imdb.images.set(1,imCount) = 1;      
                    else
                        imdb.images.set(1,imCount) = 2;
                    end
                    imdb.images.label(1,imCount) = i-2;  
                    imCount = imCount+1;
                end
            end
        end
    end
%end

    % -------------------------------------------------------------------------
    % Part 4.2: initialize a CNN architecture
    % -------------------------------------------------------------------------

    netOld = load('D:\vision\assign_3\imagenet-caffe-alex.mat');
    % newNet can be applied directly in vl_simplenn()
    net = netOld;
    for i = 1: numel(netOld.layers)
        if strcmp(netOld.layers{1,i}.type,'conv')
            net.layers{1,i} = rmfield(net.layers{1,i},'weights');
            net.layers{1,i}.filters = netOld.layers{1,i}.weights{1,1};
            net.layers{1,i}.biases = netOld.layers{1,i}.weights{1,2};
        end
    end


% -------------------------------------------------------------------------
% Training the SVM model

% -------------------------------------------------------------------------
% Take the average image out
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;

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
    im = 256 * (im - imageMean) ;
    % run the CNN
    res = vl_simplenn(net, im) ;
	featureV = res(20).x;
    [x y nFeatureV] = size(featureV);
    featureV = reshape(featureV,nFeatureV,1);
    training_instance_matrix = [training_instance_matrix featureV];
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
    featureV = resTest(20).x;
    [x y nFeatureV] = size(featureV);
    testing_instance_matrix(i,:) = (reshape(featureV,nFeatureV,1))';
end


%%Prediction Now
[predicted_label, accuracy, decision_values] = svmpredict(double(testing_label_vector), double(testing_instance_matrix), model)

[predicted_label_train, accuracy_train, decision_value_train] = svmpredict(double(testing_label_vector), double(testing_instance_matrix'), model)