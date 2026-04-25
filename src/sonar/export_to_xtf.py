import numpy as np
import ctypes, datetime, pyxtf

from ..dataset import Dataset
from .utils import get_sample_dtype, get_sample_format


def export_to_xtf(dataset: Dataset, sonar_name: str, sample_dtype: str = "uint8"):
    # Source: https://github.com/oysstu/pyxtf/blob/master/examples/write_xtf.py

    timestamps = sorted(dataset.sonar.keys())
    sonar_data = np.load(dataset.sonar_file)['data']

    sample_datatype = get_sample_dtype(sample_dtype)
    sample_bytes = np.dtype(sample_datatype).itemsize
    sample_format = get_sample_format(sample_dtype)

    # Initialize file header
    fh = pyxtf.XTFFileHeader()
    fh.SonarName = sonar_name.encode('utf-8')
    fh.SonarType = pyxtf.XTFSonarType.generic_sonar
    fh.NavUnits = pyxtf.XTFNavUnits.latlon.value
    fh.NumberOfSonarChannels = 2

    # Port chaninfo
    fh.ChanInfo[0].TypeOfChannel = pyxtf.XTFChannelType.port.value
    fh.ChanInfo[0].SubChannelNumber = 0
    fh.ChanInfo[0].BytesPerSample = sample_bytes
    fh.ChanInfo[0].SampleFormat = sample_format
    fh.ChanInfo[0].Frequency = dataset.sonar[timestamps[0]].frequency // 1000

    # Stbd chaninfo
    fh.ChanInfo[1].TypeOfChannel = pyxtf.XTFChannelType.stbd.value
    fh.ChanInfo[1].SubChannelNumber = 1
    fh.ChanInfo[1].BytesPerSample = sample_bytes
    fh.ChanInfo[1].SampleFormat = sample_format
    fh.ChanInfo[1].Frequency = dataset.sonar[timestamps[0]].frequency // 1000

    pings = []
    for i, timestamp in enumerate(timestamps):
        sidescan = dataset.sonar[timestamp]
        port, stbd = sonar_data[sidescan.ping_idx, :sidescan.num_samples], sonar_data[sidescan.ping_idx, sidescan.num_samples:]

        p = pyxtf.XTFPingHeader()
        p.HeaderType = pyxtf.XTFHeaderType.sonar.value
        p.PingNumber = i
        p.NumChansToFollow = 2

        t = datetime.datetime.fromtimestamp(timestamp / 1e9)
        p.Year = t.year
        p.Month = t.month
        p.Day = t.day
        p.Hour = t.hour
        p.Minute = t.minute
        p.Second = t.second
        p.HSeconds = int(t.microsecond / 1e4)
        p.FixTimeHour = t.hour
        p.FixTimeMinute = t.minute
        p.FixTimeSecond = t.second
        p.FixTimeHsecond = int(t.microsecond / 1e4)

        p.SoundVelocity = sidescan.speed_of_sound
        p.SensorSpeed = sidescan.navigation.speed

        # Spatial Positioning
        p.SensorXcoordinate = sidescan.navigation.longitude
        p.SensorYcoordinate = sidescan.navigation.latitude
        p.SensorDepth = sidescan.navigation.pose.z
        p.SensorPrimaryAltitude = sidescan.navigation.altitude

        # Attitude
        p.SensorPitch = np.rad2deg(sidescan.navigation.roll)
        p.SensorRoll = np.rad2deg(sidescan.navigation.pitch)
        p.SensorHeading = np.rad2deg(sidescan.navigation.yaw)

        # Setup ping chan headers
        c = (pyxtf.XTFPingChanHeader(), pyxtf.XTFPingChanHeader())

        c[0].ChannelNumber = 0
        c[0].SlantRange = sidescan.slant_range
        c[0].Frequency = sidescan.frequency // 1000
        c[0].NumSamples = len(port)

        c[1].ChannelNumber = 1
        c[1].SlantRange = sidescan.slant_range
        c[1].Frequency = sidescan.frequency // 1000
        c[1].NumSamples = len(stbd)

        p.ping_chan_headers = c
        p.data = [port, stbd]

        # Set packet size
        sz = ctypes.sizeof(pyxtf.XTFPingHeader)
        sz += ctypes.sizeof(pyxtf.XTFPingChanHeader) * 2
        sz += (len(port) + len(stbd)) * sample_bytes
        p.NumBytesThisRecord = sz

        pings.append(p)

    with open(dataset.sonar_xtf, 'wb') as f:
        f.write(fh.to_bytes())
        for p in pings:
            f.write(p.to_bytes())