-- built in modules
require 'paths'
require 'image'
require 'math'
local gm = require 'graphicsmagick'

-- custom modules
require './_util.lua'


-- List all files in a directory
function dirLookup(dir)
  local ret = {}
   local p = io.popen('find "'..dir..'" -type f')
   for file in p:lines() do
       table.insert(ret, file)
   end
   return ret
end

-- convert a bpm file at a given path to png. All files will be stored
-- in dest_folder with the same name and .png extenstion
function convert_to_png(image_path, dest_folder)
  local image_name = image_path:split('/')[#image_path:split('/')]
  image_name = image_name:gsub('bmp', 'png')
  local new_image_path = dest_folder .. '/' .. image_name

  gm.convert{
    input= image_path,
    output=new_image_path,
    quality=100,
    size= "30x30",
    verbose=true}
end


-- scales an image to a size
-- bullshit.
function scale_image_to(image_src, w, h)
  local i = image.load(image_src)
  local i_scaled = image.scale(i, w .. 'x' .. h)
  image.save(image_src, i_scaled)
end


-- pads an image to a size
-- cool
function pad_image_to(image_src, r, c)
  local i = image.load(image_src)
  local i_size = i:size()

  local i_row = i:size()[2]
  local i_col = i:size()[3]

  local extra_row = math.floor((r-i_row)/2)
  local extra_col = math.floor((c-i_col)/2)

  local res = torch.ones(1, r, c)

  for R=1, i_row do
    for C=1, i_col do
      res[1][R + extra_row][C + extra_col] = i[1][R][C]
    end
  end

  image.save(image_src, res)
end

-- extract labels based on Modares files
function extract_labels(images_dir)
  local files = dirLookup(images_dir)
  local labels = torch.Tensor(#files)

  for i=1, #files do
    labels[i] = tonumber(files[i]:split('/')[3]:split("_")[1])
  end

  return labels
end

-- extract labels based on main dataset
function extract_labels_main(images_dir)
  local files = dirLookup(images_dir)
  local labels = torch.Tensor(#files)

  for i=1, #files do
    local lbl = files[i]:split('/')[2]:split("_")[2]
    labels[i] = tonumber(lbl:sub(1, -5)) + 1
  end

  return labels
end

-- store store all labels and images of a directory in a binary file
function store_binary(images_dir, W, H, name)
  local files = dirLookup(images_dir)
  local all = {}
  local classes = {}

  all.data = torch.Tensor(#files, 1, W, H)
  all.label = extract_labels_main(images_dir)

  for i = 1, #files  do
    print('processing/saving data ' .. i .. '[' .. files[i] .. ']')
    all.data[i] = image.loadPNG(files[i])
  end

  torch.save(name, all)
end


function run(src_folder, dest_folder, binary_name)
  os.execute('mkdir ' .. dest_folder)
  local images_dirs
  local W = 30
  local H = 30

  -- convert all images to png
  images_dirs = dirLookup(src_folder)
  for i=1, #images_dirs do
    convert_to_png(images_dirs[i], dest_folder)
  end


  -- scale them all to 30x30
  images_dirs = dirLookup(dest_folder)
  for i=1, #images_dirs do
    pad_image_to(images_dirs[i], W, H)
  end

  -- store a binary version
  store_binary(dest_folder, W, H, binary_name)
end

cmd = torch.CmdLine()
cmd:text()
cmd:text('Persian OCR preprocessing')
cmd:text('Example:')
cmd:text('$> th convert.images.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--src', './PDB-Test' , 'source folder to read images from')
cmd:option('--dest','./PDB-Test-PNG', 'destination folder to save png images to')
cmd:option('--bin', 'PDB_Test.bin', 'name of the binary file to save')

opt = cmd:parse(arg or {})

run(opt.src, opt.dest, opt.bin)
