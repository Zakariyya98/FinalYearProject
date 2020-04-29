#Imported Modules
import cv2

#Cropping of Images which will aid in the Data Augementation 
def imgcrop(picture, name):
    
    crop = picture[:96, :96]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(name[:-4] + 'TopLeft' + '.png', crop)
    
    crop = picture[:96, 16:112]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(name[:-4] + 'TopCentre' + '.png', crop)
    
    crop = picture[:96, 32:]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(name[:-4] + 'TopRight' + '.png', crop)
    
    crop = picture[16:112, :96]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(name[:-4] + 'CentreLeft' + '.png', crop)
    
    crop = picture[16:112, 16:112]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(name[:-4] + 'CentreCentre' + '.png', crop)
    
    crop = picture[16:112, 32:]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(name[:-4] + 'CentrerRight' + '.png', crop)
    
    crop = picture[32:, :96]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(name[:-4] + 'BottomLeft' + '.png', crop)
    
    crop = picture[32:, 16:112]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(name[:-4] + 'BottomCentre' + '.png', crop)
    
    crop = picture[32:, 32:]
    crop = cv2.resize(crop, (128, 128))
    cv2.imwrite(name[:-4] + 'BottomRight' + '.png', crop)