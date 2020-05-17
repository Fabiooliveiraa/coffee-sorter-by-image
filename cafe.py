import cv2
import numpy as np
#imports para PICamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from matplotlib import pyplot as plt


def histogramaRGB(img):
    #Separa os canais
    canais = cv2.split(img)
    cores = ("r", "g", "b")
    fig = plt.figure()
    plt.title("'Histograma Colorido")
    plt.xlabel("Intensidade")
    plt.ylabel("Número de Pixels")
    for (canal, cor) in zip(canais, cores):
        #Este loop executa 3 vezes, uma para cada canal
        hist = cv2.calcHist([canal], [0], None, [256], [0, 256])
        plt.plot(hist, color = cor)
        #plt.xlim([100, 200])
        plt.xlim([0,255])
        plt.ylim([0,3000])
        #plt.xlim([50,150])
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return data

def histogramaPB(img):
    #Separa os canais
    fig = plt.figure()
    plt.title("'Histograma PB")
    plt.xlabel("Intensidade")
    plt.ylabel("Número de Pixels")
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    #plt.xlim([100, 200])
    plt.xlim([0,255])
    plt.ylim([0,3000])
    #plt.xlim([50,150])
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return data

def histogramaHSV(img):
    #Separa os canais
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    canais = cv2.split(hsv)
    cores = ("m", "c", "y")
    fig = plt.figure()
    plt.title("'Histograma HSV")
    plt.xlabel("Intensidade")
    plt.ylabel("Número de Pixels")
    for (canal, cor) in zip(canais, cores):
        #Este loop executa 3 vezes, uma para cada canal
        hist = cv2.calcHist([canal], [0], None, [256], [0, 256])
        plt.plot(hist, color = cor)
        plt.xlim([0, 256])
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return data

def imprime_cor(img): #imagem deve estar em formato RGB
    blue = 0
    green = 0
    red = 0
    for p in np.ravel(img[::4,::4,0]):
       blue+=p
   
    for p in np.ravel(img[::4,::4,1]):
       green+=p
    
    for p in np.ravel(img[::4,::4,2]):
       red+=p
    red   = red/1000
    green = green/1000
    blue  = blue/1000

    print('red:',red,' green:',green,' blue:',blue, end='')
    if red>15 and red<35 and green>20 and green<38: 
       print('GRAO PRETO')
       return 'PRETO'
    if red>50 and red<90 and green>50 and green<90: 
       print('GRAO VERDE')
       return 'VERDE'
    if red>35 and red<80 and green>20 and green<60: 
       print('GRAO ARDIDO')
       return 'ARDIDO'

def texto(img, texto, coord, fonte = cv2.FONT_HERSHEY_SIMPLEX, cor=(0,0,255), tamanho=0.7, thickness=2):
    textSize, baseline = cv2.getTextSize(texto, fonte, tamanho, thickness);
    cor_background = 0
    if type(cor)==int: # se não for colorida a imagem
        cor_background=255-cor
    else:
        cor_background=(255-cor[0],255-cor[1],255-cor[2])
    #print(cor_background)
    cv2.rectangle(img, (coord[0], coord[1]-textSize[1]-3), (coord[0]+textSize[0], coord[1]+textSize[1]-baseline), cor_background, -1)
    #cv2.putText(img, texto, coord, fonte, tamanho, cor_background, thickness+1, cv2.LINE_AA)
    cv2.putText(img, texto, coord, fonte, tamanho, cor, thickness, cv2.LINE_AA)
    return img

if __name__ == '__main__':

    camera = PiCamera()
    rawCapture = PiRGBArray(camera)  # , size=(320, 240))
    time.sleep(0.1)
    roi = (600,350,700,450) # cria variável roi
    print('>> ROI:', roi)
    for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True): # para uso com Pi Camera
        frame = frame.array
        #aplica ROI em todos os frames do primeiro em diante
        frame = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        img_width, img_height = frame.shape[1], frame.shape[0] 
        #print('Shape:', img_width, img_height)
        try:    # Lookout for a keyboardInterrupt to stop the script
                frameRGB = frame[:,:,::-1] # inverte BGR para RGB
                
                # find the colors within the specified boundaries and apply the mask
                #mask = cv2.inRange(frameRGB, (0,0,0), (200,200,200))
                frameBP = cv2.cvtColor(frameRGB, cv2.COLOR_BGR2GRAY) 
                (T, bin) = cv2.threshold(frameBP, 100, 255, cv2.THRESH_BINARY_INV)
                mask = bin
                frame_sub = cv2.bitwise_and(frameRGB, frameRGB, mask = mask)
                #mascara = cv2.cvtColor(cv2.merge([mask,mask,mask]), cv2.COLOR_BGR2GRAY)
                mascara = cv2.merge([mask,mask,mask])
                
                cor_encontrada = imprime_cor(frame_sub.copy())
                histRGB = histogramaRGB(frame_sub.copy())
                histRGB = cv2.resize(histRGB.copy(), (320,240), interpolation = cv2.INTER_AREA)
                histPB = histogramaPB(frame_sub.copy())
                histPB = cv2.resize(histPB.copy(), (320,240), interpolation = cv2.INTER_AREA)
                #histHSV = histogramaHSV(frame_sub.copy())
                #histHSV = cv2.resize(histHSV.copy(), (320,240), interpolation = cv2.INTER_AREA)
                size = (histRGB.shape[1], histRGB.shape[0])
                frameRGB_res = cv2.resize(frameRGB, size, interpolation = cv2.INTER_AREA)
                frame_sub_res = cv2.resize(frame_sub, size, interpolation = cv2.INTER_AREA)
                mascara_res = cv2.resize(mascara, size, interpolation = cv2.INTER_AREA)
                
                frameBP = cv2.cvtColor(frameBP, cv2.COLOR_GRAY2RGB) 
                frameBP = cv2.resize(frameBP, size, interpolation = cv2.INTER_AREA)
                
                #print([frameRGB_res.shape, frame_sub_res.shape, frameBP.shape]),
                #print([histRGB.shape, histHSV.shape, mascara_res.shape]),
		
                frameRGB_res = texto(frameRGB_res, cor_encontrada, (10,30))

                join = np.vstack([
                    np.hstack([frameRGB_res, frame_sub_res, frameBP]),
                    np.hstack([histRGB, histPB, mascara_res]),
                ])

                

                window_name = "Cafe"
                #cv2.namedWindow(window_name, flags=cv2.WND_PROP_FULLSCREEN);
                #cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                #cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow(window_name, join) #converte para BGR para mostrar
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break
        except KeyboardInterrupt:
                # vc.release() # só usado com camera USB
                cv2.destroyAllWindows()
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0) # para uso com Pi Camera


