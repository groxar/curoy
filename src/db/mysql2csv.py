from __future__ import division
import MySQLdb
import datetime
import calendar

fmt = '%Y-%m-%d %H:%M:%S'

def getDate(datetimestring):
    return datetime.datetime.strptime(datetimestring, fmt)

def getHourlyDateTimes(previousDate, date):
    totalSeconds = (date - previousDate).total_seconds()
    totalHours = int(totalSeconds / 3600)

    allDates = []
    for i in range(0, totalHours):
        allDates.insert(0, date - i * datetime.timedelta(0, 3600))
    return allDates

def getDateFeatures(date):
    dayofweek = calendar.weekday(date.year, date.month, date.day)
    return {'hour': str(date.hour + date.minute / 60. + date.second / 3600.),'day': str(date.day), 'dayofweek': str(dayofweek), 'month': str(date.month)}

def readDevice(deviceID, db, path):
    cur = db.cursor()
    cur.execute("SELECT startTime, measurementValue FROM measurements WHERE deviceId='%s'" % deviceID)
    filex = open(path + str(deviceID) + 'x.txt', 'w')
    filey = open(path + str(deviceID) + 'y.txt', 'w')
    previousDate = None
    previousMeasurement = None
    for row in cur.fetchall():
        date = getDate(str(row[0]))
        measurement = float(row[1])
        if previousDate is None:
            previousDate = date - datetime.timedelta(0, 3600)
        if previousMeasurement is None:
            previousMeasurement = measurement
        allHourlyDates = getHourlyDateTimes(previousDate, date)
        if(not len(allHourlyDates) < 1):
            delta = (measurement - previousMeasurement) / len(allHourlyDates)
            for hourlyDate in allHourlyDates:    
                dateFeatures = getDateFeatures(hourlyDate);
                filex.write(' ' + dateFeatures['hour'] + ' ' + dateFeatures['day'] + ' ' + dateFeatures['dayofweek'] + ' ' + dateFeatures['month'] + '\n')
                filey.write(' ' + str(delta) + '\n')
            previousDate = date
            previousMeasurement = measurement
    filex.close()
    filey.close()
    cur.close()



    
db = MySQLdb.connect(host="129.13.170.32", # your host, usually localhost
                     user="ding", # your username
                      passwd="vEPbn4MW4bPXwDS7", # your password
                      db="ding") # name of the data base

cur = db.cursor() 

cur.execute("SELECT * FROM devices")

path = 'C:/Users/Matthias/Desktop/data/'
file = open(path + "devices.txt", 'w')
devices = []
for row in cur.fetchall() :
    file.write(str(row[0]))
    file.write('\n')
    devices.append(str(row[0]))
cur.close()
file.close

i = 0
n = len(devices)
for device in devices:
    print str(i / n) + "%"
    i = i + 1
    readDevice(device, db, path)

print "finished"
