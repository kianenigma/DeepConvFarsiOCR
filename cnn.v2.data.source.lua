require 'dp'

-- create classes as a table - needed in dp
local classes = {}
-- local classes = {
--   "ا",
--   "ب",
--   "پ",
--   "ت",
--   "ث",
--   "ج",
--   "چ",
--   "ح",
--   "خ",
--   "ر",
--   "ز",
--   "ژ",
--   "س",
--   "ش",
--   "ص",
--   "ض",
--   "ط",
--   "ظ",
--   "ع",
--   "غ",
--   "ف",
--   "ق",
--   "ک",
--   "گ",
--   "ل",
--   "م",
--   "ن",
--   "و",
--   "ه",
--   "ی",
--   "ی",
--   "ی",
--   "ی",
--   "ی",
--   "ی",
--   "ی",
--   "ی",
--   "ی",
-- }

for i=1, 36 do
  classes[i] = i
end

-- 1 params
local validRatio = .5
local train_bin = './PDB_Train.bin'
local test_bin = './PDB_Test.bin'

-- 2 load train data
local train = torch.load(train_bin)
local test = torch.load(test_bin)

-- 3 divide test and valid set
nTotal = test.data:size()[1]
local nValid = math.floor(nTotal*validRatio)
local nTest = nTotal - nValid

-- 4 create data set
local trainInput = dp.ImageView('bchw', train.data)
local trainTarget = dp.ClassView('b', train.label)

local testInput = dp.ImageView('bchw', test.data:narrow(1,1, nTest))
local testTarget = dp.ClassView('b', test.label:narrow(1,1, nTest))

local validInput = dp.ImageView('bchw', test.data:narrow(1, nTest+1, nValid))
local validTarget = dp.ClassView('b', test.label:narrow(1, nTest+1, nValid))

trainTarget:setClasses(classes)
validTarget:setClasses(classes)
testTarget:setClasses(classes)

-- wrap in dataset
local trainDS = dp.DataSet{inputs=trainInput, targets=trainTarget, which_set='train'}
local validDS = dp.DataSet{inputs=validInput, targets=validTarget, which_set='valid'}
local testDS = dp.DataSet{inputs=testInput, targets=testTarget, which_set='test'}

-- wrap in datasource
local ds = dp.DataSource{train_set= trainDS, valid_set=validDS, test_set=testDS}
ds:classes(classes)

return ds
