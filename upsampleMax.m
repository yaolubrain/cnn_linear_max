function up = upsampleMax(convolvedFeatures,poolIdx,poolDim)
up = zeros(size(convolvedFeatures)*poolDim);
up(poolIdx) = convolvedFeatures(:);
