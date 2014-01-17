testImages = loadMNISTImages('../common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageSize,imageSize,[]);
testLabels = loadMNISTLabels('../common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10
testSize = length(testImages);
preds = zeros(testSize,1);

mbSize = size(testImages,3);
% Feedforward

o1 = zeros(convDim1,convDim1,filterNum1,mbSize);                
for i = 1:filterNum1
    o1(:,:,i,:) = convn(testImages,rot90(Wc1(:,:,i),2),'valid') + bc1(i);                   
end        
o1Pooled = zeros(outputDim1,outputDim1,filterNum1,mbSize);
o1PoolIdx = zeros(outputDim1^2,filterNum1,mbSize);
for i = 1:mbSize
    for j = 1:filterNum1
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
        [o2Pooled(:,:,j,i) o2PoolIdx(:,j,i)] = MaxPooling(o2(:,:,j,i),[poolDim2 poolDim2]);
    end
end
o2PooledVec = reshape(o2Pooled,[],mbSize);
o3 = Wd*o2PooledVec + repmat(bd,[1,mbSize]);
    
[~,preds] = max(o3);        

acc = sum(preds'==testLabels)/length(preds);
fprintf('Accuracy is %f\n',acc);