# DeepConvFarsiOCR
A Deep Convolutional Approach Toward Farsi Character Recognition. Used for both machine printed and handwritten datasets


## Usage

#### Step1: Data
  - Download all images from this link
  - Extract them to the home directory. After this, you should see  `PDB-Train` and `PDB-Test` folders in your home directory.
  - ensure that `run()` function in `1_convert_images.lua` is not commented.
  - execute `run()` from `1_convert_images.lua` twice, with the following lines in the beginning of `run()`:
  ```
  local src_folder = './PDB-Test'
  local dest_folder = './PDB-Test-PNG'
  local binary_name = 'PDB_Test.bin'
  ```

  and

  ```
  local src_folder = './PDB-Train'
  local dest_folder = './PDB-Train-PNG'
  local binary_name = 'PDB_Train.bin'
  ```

  Two new folders, `PDB-Train-PNG` and `PDB-Test-PNG` should be created. Two binary files should also be created

#### Step2: Train
  - run `th cnn.v2.lua --progress`. See the source file for more commands and options.
