import os, fnmatch
import matplotlib.pyplot as plt

def dirfiles(img_path,img_ext):
    img_names = fnmatch.filter(sorted(os.listdir(img_path)),img_ext)
    if '.DS_Store' in img_names:
        img_names.remove('.DS_Store')
    return img_names

def num2fixstr(x,d):
    st = '%0*d' % (d,x)
    return st

def imageload(prefix,num_class,digits_class,num_img,digits_img,echo='off'):
  st   = prefix + num2fixstr(num_class,digits_class) + '_' + num2fixstr(num_img,digits_img) + '.png'
  if echo == 'on':
    print('loading image '+st+'...')
  img    = plt.imread(st)
  return img

