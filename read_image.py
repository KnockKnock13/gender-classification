import numpy as np
import Image

#train_data = np.empty((760, 100, 100))
#train_lable = np.zeros((760, 2))
#test_data = np.empty((100, 100, 100))
#test_lable = np.zeros((100, 2))

def path(i):
  if i>9:
    if i>99:
      return '0'+str(i)
    else:
      return '00'+str(i)
  else:
    return '000'+str(i)

def read(switch):
  i=0
  j=0
  if switch==1:
    train_data = np.empty((760, 100, 100))
    train_lable = np.zeros((760, 2))

    while i<380: #training data
      im = Image.open("image/image/Male_folder/"+path(i+1)+"/my.bmp")#male
      train_data[j,:,:] = np.array(im)
      train_data[j,:,:] = train_data[j,:,:]/255.0
      s = sum(sum(train_data[j,:,:]))/10000
      train_data[j,:,:]=train_data[j,:,:]-s
      temp = train_data[j,:,:]*train_data[j,:,:]
      d = sum(sum(temp))
   # train_data[j,:,:] = train_data[j,:,:]/d
   # print(d)
    #print(s)
      train_lable[j,0] = 1
      j = j + 1
      im = Image.open("image/image/Female_folder/"+path(i+1)+"/fy.bmp")#female
      train_data[j, :, :] = np.array(im) 
      train_data[j,:,:] = train_data[j,:,:]/255.0
      s = sum(sum(train_data[j,:,:]))/10000
      train_data[j,:,:]=train_data[j,:,:]-s
      temp = train_data[j,:,:]*train_data[j,:,:]
      d = sum(sum(temp))
    #train_data[j,:,:] = train_data[j,:,:]/d 
      train_lable[j, 1] = 1
      i = i + 1 
      j = j + 1
  else:
    i=380
  test_data = np.empty((100, 100, 100))
  test_lable = np.zeros((100, 2))

  while i<50+380:#test data
    im = Image.open("image/image/Male_folder/"+path(i+1)+"/my.bmp")#male
    test_data[i-380, :, :] = np.array(im)
    test_data[i-380,:,:] = test_data[i-380,:,:]/255.0
    s = sum(sum(test_data[i-380,:,:]))/10000
    test_data[i-380,:,:]=test_data[i-380,:,:]-s
    temp = test_data[i-380,:,:]*test_data[i-380,:,:]
    d = sum(sum(temp))
    #train_data[i-380,:,:] = train_data[i-380,:,:]/d
    test_lable[i-380, 0] = 1
    i = i + 1
  j=380
  i=0
  while i<50:#female
    im = Image.open("image/image/Female_folder/"+path(j+1)+"/fy.bmp")#female
    test_data[50+i, :, :] = np.array(im)
    test_data[50+i,:,:] = test_data[50+i,:,:]/255.0
    s = sum(sum(test_data[50+i,:,:]))/10000
    test_data[50+i,:,:]=test_data[50+i,:,:]-s
    temp = test_data[50+i,:,:]*test_data[50+i,:,:]
    d = sum(sum(temp))
    #train_data[50+i,:,:] = train_data[50+i,:,:]/d
    test_lable[50+i, 1] = 1
    i = i + 1
    j = j + 1
  if switch==1:
    return train_data,train_lable,test_data,test_lable
  else: 
    return test_data, test_lable

def debug():
 test_data,test_lable = read(2)
 print(test_lable[10, :], test_data.shape)
 image = Image.fromarray(test_data[10, :, :])
 image.show()
 print(test_data[10,:,:])


if __name__ == '__main__':
  debug()

