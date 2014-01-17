clear all

% Hyper-parameters
imageSize = 28;
filterNum1 = 10;  
filterSize1 = 5;  
poolDim1 = 2;
filterNum2 = 10;  
filterSize2 = 5;  
poolDim2 = 2;
convDim1 = imageSize - filterSize1 + 1;
outputDim1 = convDim1/poolDim1;
convDim2 = outputDim1 - filterSize2 + 1;
outputDim2 = convDim2/poolDim2;
classNum = 10;
sampleSize = 60000; 
mbSize = 50; % mini-batch sample size
r = 0.05;    % learning rate
wdr = 0.00; % weight decay rate
mom = 0.5;   % momentum
epochNum = 100;

% Load data
addpath('../common');
images = loadMNISTImages('../common/train-images-idx3-ubyte');
images = reshape(images,imageSize,imageSize,[]);
labels = loadMNISTLabels('../common/train-labels-idx1-ubyte');
labels(labels==0) = 10;
labelMat = full(sparse(labels,1:length(labels),1));

% Initialize parameters
Wc1 = 0.1*(randn(filterSize1,filterSize1,filterNum1));
bc1 = zeros(filterNum1,1);
Wc2 = 0.1*(randn(filterSize2,filterSize2,filterNum1,filterNum2));
bc2 = zeros(filterNum2,1);
Wd = 0.01*(rand(classNum,filterNum2*outputDim2^2) - 0.5);
bd = zeros(classNum,1);

% Velocity of parameters
vel_Wc1 = zeros(size(Wc1));
vel_bc1 = zeros(size(bc1));
vel_Wc2 = zeros(size(Wc2));
vel_bc2 = zeros(size(bc2));
vel_Wd = zeros(size(Wd));
vel_bd = zeros(size(bd));

%% Training
for e = 1:epochNum
    err = 0;    
    rp = randperm(sampleSize);    
    for s = 1:mbSize:(sampleSize-mbSize+1)        
        mbImages = images(:,:,rp(s:s+mbSize-1));        
        
        % Feedforward
        o1 = zeros(convDim1,convDim1,filterNum1,mbSize);                
        for i = 1:filterNum1
            o1(:,:,i,:) = convn(mbImages,rot90(Wc1(:,:,i),2),'valid') + bc1(i);                   
        end        
        o1Pooled = zeros(outputDim1,outputDim1,filterNum1,mbSize);
        o1PoolIdx = zeros(outputDim1^2,filterNum1,mbSize);
        for i = 1:mbSize
            for j = 1:filterNum1
%                 o1Pooled(:,:,i,j) = meanPool(o1(:,:,i,j),poolDim1);    
                [o1Pooled(:,:,j,i) o1PoolIdx(:,j,i)] = MaxPooling(o1(:,:,j,i),[poolDim1 poolDim1]);
            end
        end        
        o2 = zeros(convDim2,convDim2,filterNum2,mbSize);              
        for i = 1:filterNum2
            for j = 1:filterNum1
                o2(:,:,i,:) = o2(:,:,i,:) + convn(o1Pooled(:,:,j,:),rot90(Wc2(:,:,j,i),2),'valid');
            end
            o2(:,:,i,:) = o2(:,:,i,:) + bc2(i);
        end        
        o2Pooled = zeros(outputDim2,outputDim2,filterNum2,mbSize);
        o2PoolIdx = zeros(outputDim2^2,filterNum2,mbSize);
        for i = 1:mbSize 
            for j = 1:filterNum2
%                 o2Pooled(:,:,i,j) = meanPool(o2(:,:,i,j),poolDim2);
                [o2Pooled(:,:,j,i) o2PoolIdx(:,j,i)] = MaxPooling(o2(:,:,j,i),[poolDim2 poolDim2]);
            end
        end
        o2PooledVec = reshape(o2Pooled,[],mbSize);
        o3 = Wd*o2PooledVec + repmat(bd,[1,mbSize]);
        
        % Back Propagation
        y = labelMat(:,rp(s:s+mbSize-1));                
        delta_d = o3 - y;
        delta_s2 = Wd' * delta_d;
        delta_s2 = reshape(delta_s2,outputDim2,outputDim2,filterNum2,mbSize);
        delta_c2 = zeros(convDim2,convDim2,filterNum2,mbSize);
        for i = 1:mbSize
            for j = 1:filterNum2
%                 delta_c2(:,:,j,i) = upsampleMean(delta_s2(:,:,j,i),poolDim2);
                delta_c2(:,:,j,i) = upsampleMax(delta_s2(:,:,j,i),o2PoolIdx(:,j,i),poolDim2);
            end
        end
        delta_s1 = zeros(outputDim1,outputDim1,filterNum1,mbSize);
        for i = 1:filterNum1
            for j = 1:filterNum2
                delta_s1(:,:,i,:) = delta_s1(:,:,i,:) + convn(delta_c2(:,:,j,:),Wc2(:,:,i,j),'full');
            end
        end
        delta_c1 = zeros(convDim1,convDim1,filterNum1,mbSize);
        for i = 1:mbSize
            for j = 1:filterNum1      
%                 delta_c1(:,:,j,i) = upsampleMean(delta_s1(:,:,j,i),poolDim1);
                delta_c1(:,:,j,i) = upsampleMax(delta_s1(:,:,j,i),o1PoolIdx(:,j,i),poolDim1);
            end
        end

        grad_Wd = (1/mbSize)*delta_d*o2PooledVec';
        grad_bd = zeros(size(bd));
        for i = 1:classNum
            grad_bd(i) = (1/mbSize)*sum(delta_d(i,:));
        end
        grad_Wc2 = zeros(size(Wc2));
        grad_bc2 = zeros(size(bc2));
        for i = 1:filterNum2            
            for j = 1:filterNum1
                for k = 1:mbSize
                    grad_Wc2(:,:,j,i) = grad_Wc2(:,:,j,i) + conv2(o1Pooled(:,:,j,k),rot90(delta_c2(:,:,i,k),2),'valid');
                end
                grad_Wc2(:,:,j,i) = (1/mbSize)*grad_Wc2(:,:,j,i);        
            end
            tmp_grad_bc2 = delta_c2(:,:,i,:);  
            grad_bc2(i) = (1/mbSize)*sum(tmp_grad_bc2(:));
        end
        grad_Wc1 = zeros(size(Wc1));
        grad_bc1 = zeros(size(bc1));
        for i = 1:filterNum1            
            for j = 1:mbSize
                grad_Wc1(:,:,i) = grad_Wc1(:,:,i) + conv2(mbImages(:,:,j),rot90(delta_c1(:,:,i,j),2),'valid');
            end
            grad_Wc1(:,:,i) = (1/mbSize)*grad_Wc1(:,:,i);
            tmp_grad_bc1 = delta_c1(:,:,i,:);  
            grad_bc1(i) = (1/mbSize)*sum(tmp_grad_bc1(:));
        end
                
        vel_Wd = mom*vel_Wd - r*grad_Wd - wdr*Wd;
        vel_bd = mom*vel_bd - r*grad_bd - wdr*bd;
        vel_Wc2 = mom*vel_Wc2 - r*grad_Wc2 - wdr*Wc2;
        vel_bc2 = mom*vel_bc2 - r*grad_bc2 - wdr*bc2;
        vel_Wc1 = mom*vel_Wc1 - r*grad_Wc1 - wdr*Wc1;
        vel_bc1 = mom*vel_bc1 - r*grad_bc1 - wdr*bc1;
                        
        Wd = Wd + vel_Wd;
        bd = bd + vel_bd;
        Wc2 = Wc2 + vel_Wc2;
        bc2 = bc2 + vel_bc2;
        Wc1 = Wc1 + vel_Wc1;
        bc1 = bc1 + vel_bc1;     
        
        mbErr = mean(delta_d(:).^2);
        err = err + mbErr;
    end           
    fprintf('Epoch %d Training error: %f\n',e,err);
    r = 0.98*r;
end

cnnTest;