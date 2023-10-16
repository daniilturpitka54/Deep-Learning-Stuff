import re, csv
from astropy import units as u
from astropy.coordinates import SkyCoord
filew = open('lense_predictions3.csv', 'w')
filer = open('predicted_lensesxgb6.txt', 'r')
preds = [(i.split(',')[0],i.split(',')[1]) for i in filer.readlines()]
preds = sorted(preds, key=lambda x: -float(x[1]))
preds = preds[:500]
csvwriter = csv.writer(filew, delimiter=',')
csvwriter.writerow(['Name', 'RA', 'Dec', 'Method'])
namedict = dict()
dictfile = open('namesdict.csv','r')
for i in dictfile.readlines()[1:]:
    sp = i.split(',')
    #print(type(sp[1]))
    namedict[str(sp[1])] = (float(sp[2]),float(sp[3]))
for p in preds:
    wstr = []
    i = p[0]
    #print(i)
   
    if 'GRALJ'in i:
        #HMS to degree
        name = re.search('GRALJ\d{6,}[-+]\d{6,}', i)
        wstr.append(name.group())
        exph = re.search('\d{6,}', i)
        exph = exph.group()
        expdms = re.search('[-+]\d{6,}',i)
        expdms = expdms.group()
        h = float(exph[0:2])*15
        m = float(exph[2:4])*(15/60)
        s = exph[4:6]
        secdec = exph[6:]
        s = float(s+'.'+secdec)*(15/3600)
        ras = h+m+s
        wstr.append(ras)
        #DMS to decimal
        dec = 0.0
        h = float(expdms[1:3])
        m = float(expdms[3:5])/60
        s = float(expdms[5:7])/3600
        tot = h+m+s
        secdec = expdms[7:]
        ln = len(secdec)
        if float(expdms)<0:
            tot = tot*(-1)
        c = SkyCoord(ra = ras * u.degree, dec = tot*u.degree)
        
        if abs(c.galactic.b.deg) >10:
            print(c.galactic.l.deg)
            wstr.append(tot)
            wstr.append('DT-Autoencoder')
            wstr.append(float(p[1]))
            csvwriter.writerow([str(i) for i in wstr])
        #print(i, wstr)
    else:
        sp = i.split(',')
        nm = sp[0][:14]
        try:
            c = SkyCoord(ra = namedict[nm][0]*u.degree, dec = namedict[nm][1]*u.degree)
            if abs(c.galactic.b.deg )>10:
                print(c.galactic.l.deg)
                csvwriter.writerow([nm, namedict[nm][0], namedict[nm][1], 'DT-Autoencoder',p[-1][:-2] ])
        except Exception as ex:
            wstr = []
            name = re.search('\d{6,}[-+]\d{6,}', i)
            wstr.append(name.group())
            exph = re.search('\d{6,}', i)
            exph = exph.group()
            expdms = re.search('[-+]\d{6,}',i)
            expdms = expdms.group()
            h = float(exph[0:2])*15
            m = float(exph[2:4])*(15/60)
            s = exph[4:6]
            secdec = exph[6:]
            s = float(s+'.'+secdec)*(15/3600)
            ras = h+m+s
            wstr.append(ras)
            #DMS to decimal
            dec = 0.0
            h = float(expdms[1:3])
            m = float(expdms[3:5])/60
            s = float(expdms[5:7])/3600
            tot = h+m+s
            secdec = expdms[7:]
            ln = len(secdec)
            if float(expdms)<0:
                tot = tot*(-1)
            c = SkyCoord(ra = ras * u.degree, dec = tot*u.degree)
            if abs(c.galactic.l.deg) >10:
                print(c.galactic.b.deg)
                wstr.append(tot)
                wstr.append('DT-Autoencoder')
                wstr.append(float(p[1]))
                csvwriter.writerow([str(i) for i in wstr])

filew.close()
filer.close()

        
        

#t = (11.0*15)+(38.0*15/60)+(57.0*15/3600)+(0.7*15/3600)
#print(t)
        

       