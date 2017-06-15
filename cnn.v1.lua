require 'paths'
require 'nn'
require 'cunn'
require 'image'

-- META
VALID_SIZE = 50
DIM = 30
CLASSES = 10

-- data loading
all = torch.load('modares.bin')
COUNT = all.data:size()[1]

print('Total data', COUNT)

trainset = {}
trainset.data = all.data[{{1, COUNT-VALID_SIZE}}]
trainset.label = all.label[{{1, COUNT - VALID_SIZE}}]

testset = {}
testset.data = all.data[{{COUNT-VALID_SIZE+1, COUNT}}]
testset.label = all.label[{{COUNT-VALID_SIZE+1, COUNT}}]

print('trainset size', trainset.data:size()[1])
print("testset size", testset.data:size()[1])

-- some meta data modification
setmetatable(trainset,
    {
      __index = function(t, i)
                    return {t.data[i], t.label[i]}
                end
    }
);

function trainset:size()
    return self.data:size(1)
end

net = nn.Sequential()

net:add(nn.SpatialConvolution(1, 64, 5, 5))       -- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                                -- non-linearity
net:add(nn.SpatialMaxPooling(2,2,2,2))            -- A max-pooling operation that looks at 2x2 windows and finds the max.

net:add(nn.SpatialConvolution(64, 128, 5, 5))
net:add(nn.ReLU())                                -- non-linearity
net:add(nn.SpatialMaxPooling(2,2,2,2))

net:add(nn.View(128*4*4))                         -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5

net:add(nn.Linear(128*4*4, 256))                  -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                                -- non-linearity

net:add(nn.Linear(256, 128))
net:add(nn.ReLU())                                -- non-linearity

net:add(nn.Linear(128, CLASSES))                  -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                          -- converts the output to a log-probability. Useful for classification problems

net = net:cuda()

print('### NETWORK:\n' .. net:__tostring());

start = os.time()

-- Loss Criteria
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

-- Train
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 100

trainer:train(trainset)

print('took me ' .. os.time() - start .. 'ms  to finish')

-- prepare test data
testset.data = testset.data:cuda()   -- convert from Byte tensor to Double tensor

-- evaluate
correct = 0
for i=1,VALID_SIZE do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end
print(correct, 100*correct/VALID_SIZE .. ' % ')
-- torch.save('cnn-full-250iter-conv32-64-croppedData', net)
