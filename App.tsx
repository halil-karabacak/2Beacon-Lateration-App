import React, { useEffect, useState, useRef } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  PermissionsAndroid,
  Platform,
  FlatList,
  Switch,
  Button,
  TextInput,
  StyleSheet,
} from 'react-native';
import { BleManager } from 'react-native-ble-plx';
import Svg, { Circle } from 'react-native-svg';
import * as math from 'mathjs'; // for matrix operations in Kalman Filter

// ***** BLE Manager *****
const manager = new BleManager();

// ***** Kalman Filter for 1D corridor (t + velocity) *****
class KalmanFilter1D {
  constructor(dt = 0.1) {
    // State: x = [ t, tDot ]^T
    // Let's assume an initial state of t=0.5 (mid-corridor), zero velocity
    this.x = math.matrix([[0.5], [0]]);

    // Covariance
    this.P = math.multiply(math.identity(2), 1);

    // Process noise (Q) - tweak as needed
    this.Q = math.matrix([
      [0.001, 0],
      [0, 0.001],
    ]);

    // Measurement noise (R) - tweak as needed
    this.R = math.matrix([[0.01]]);

    // Time step
    this.dt = dt;

    // State transition matrix F (constant velocity model)
    this.F = math.matrix([
      [1, this.dt],
      [0, 1],
    ]);

    // Measurement matrix H (we measure only t)
    this.H = math.matrix([[1, 0]]);
  }

  // Predict step
  predict() {
    // x = F x
    this.x = math.multiply(this.F, this.x);

    // P = F P F^T + Q
    let FP = math.multiply(this.F, this.P);
    let FPFt = math.multiply(FP, math.transpose(this.F));
    this.P = math.add(FPFt, this.Q);
  }

  // Update step (z is the measured t)
  update(zVal) {
    const z = math.matrix([[zVal]]);
    // y = z - H x
    const y = math.subtract(z, math.multiply(this.H, this.x));
    // S = H P H^T + R
    const S = math.add(
      math.multiply(math.multiply(this.H, this.P), math.transpose(this.H)),
      this.R
    );
    // K = P H^T S^-1
    const K = math.multiply(
      math.multiply(this.P, math.transpose(this.H)),
      math.inv(S)
    );
    // x = x + K y
    this.x = math.add(this.x, math.multiply(K, y));
    // P = (I - K H) P
    const I = math.identity(2);
    const KH = math.multiply(K, this.H);
    this.P = math.multiply(math.subtract(I, KH), this.P);

    // Optionally clamp t in [0,1]
    const t_clamped = Math.max(0, Math.min(1, this.x.get([0, 0])));
    this.x.set([0, 0], t_clamped);
  }

  // Single step: predict + update
  processMeasurement(measuredT) {
    this.predict();
    this.update(measuredT);
  }

  get currentT() {
    return this.x.get([0, 0]);
  }
  
  get currentVelocity() {
    return this.x.get([1, 0]);
  }
}

// ***** Main App *****
const App = () => {
  const [devices, setDevices] = useState([]);
  const [selectedDevices, setSelectedDevices] = useState({});
  const [rssiHistory, setRssiHistory] = useState({});

  const SMOOTHING_WINDOW = 5;

  // Beacon positions (assuming a corridor on the same y-level)
  // x1 < x2 => corridor is from beacon1.x to beacon2.x
  const [beaconPositions, setBeaconPositions] = useState({
    beacon1: { x: 100, y: 100 } ,
    beacon2: { x: 300, y: 100 },
  });
  const [corridorLength, setCorridorLength] = useState(200); // initial guess

  // RSSI -> Distance settings
  const [settings, setSettings] = useState({ A: -59, n: 2 });

  // The user’s position in (x, y), updated from the Kalman filter each time
  const [userPosition, setUserPosition] = useState({ x: 0, y: 0 });

  // Kalman filter ref
  const kalmanFilterRef = useRef(null);

  // Are we scanning/positioning?
  const [isCalculating, setIsCalculating] = useState(false);

  // ***** Permission and Scanning *****
  useEffect(() => {
    requestPermissions().then(() => {
      // Once permitted, start scanning if desired
      startScanning();
    });

    // Initialize corridor length from the 2 beacon positions
    updateCorridorLength(beaconPositions);

    // Initialize Kalman filter
    if (!kalmanFilterRef.current) {
      kalmanFilterRef.current = new KalmanFilter1D(0.1); // 0.1s time step
    }

    return () => {
      manager.stopDeviceScan();
    };
  }, []);

  const requestPermissions = async () => {
    if (Platform.OS === 'android') {
      if (Platform.Version >= 31) {
        await PermissionsAndroid.requestMultiple([
          PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
          PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        ]);
      } else {
        await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION
        );
      }
    }
  };

  const startScanning = () => {
    manager.startDeviceScan(null, { allowDuplicates: true }, (error, device) => {
      if (error) {
        console.error('BLE scan error:', error);
        return;
      }

      if (device) {
        // Keep a list of discovered devices (id, name, rssi)
        setDevices((prev) => {
          const existing = prev.find((d) => d.id === device.id);
          if (!existing) {
            return [
              ...prev,
              {
                id: device.id,
                name: device.name || 'Unknown',
                rssi: device.rssi ?? 0,
              },
            ];
          }
          return prev.map((d) =>
            d.id === device.id ? { ...d, rssi: device.rssi ?? 0 } : d
          );
        });

        // Maintain RSSI history for smoothing
        setRssiHistory((prev) => {
          const history = prev[device.id]?.history || [];
          const updatedHistory = [...history, device.rssi].slice(-SMOOTHING_WINDOW);
          const smoothedRssi =
            updatedHistory.reduce((sum, value) => sum + value, 0) / updatedHistory.length;

          return {
            ...prev,
            [device.id]: { history: updatedHistory, smoothedRssi },
          };
        });
      }
    });
  };

  // Update the corridor length if the beacons change
  const updateCorridorLength = (beacons) => {
    const dx = beacons.beacon2.x - beacons.beacon1.x;
    const dy = beacons.beacon2.y - beacons.beacon1.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    setCorridorLength(dist);
  };

  useEffect(() => {
    // Recompute corridor length if beaconPositions changes
    updateCorridorLength(beaconPositions);
  }, [beaconPositions]);

  // ***** Utility Functions *****

  // Simple path-loss model
  const calculateDistance = (rssi, A, n) => {
    // d = 10 ^ ((RSSI - A) / (-10n))
    return Math.pow(10, (rssi - A) / (-10 * n));
  };

  /**
   * computeT:
   * Given distances d1, d2 to two beacons separated by corridorLength = D,
   * we solve the least squares for t in [0,1]:
   *
   * We want: (t*D ~ d1) and ((1-t)*D ~ d2)
   * Minimizing squared error => t = (d1 + D - d2) / (2D).
   * Then clamp [0,1].
   */
  const computeT = (d1, d2, D) => {
    if (D <= 0) return 0;
    const rawT = (d1 + D - d2) / (2 * D);
    return Math.max(0, Math.min(1, rawT));
  };

  /**
   * getPositionOnCorridor:
   * Convert a corridor parameter t in [0,1] to an (x,y)
   * between beacon1 and beacon2.
   */
  const getPositionOnCorridor = (t, p1, p2) => {
    return {
      x: p1.x + t * (p2.x - p1.x),
      y: p1.y + t * (p2.y - p1.y),
    };
  };

  // ***** Calculate Position Each Frame *****
  const calculatePosition = () => {
    // Only proceed if we have 2 selected devices
    const selected = devices.filter((device) => selectedDevices[device.id]);
    if (selected.length === 2 && corridorLength > 0) {
      const [device1, device2] = selected;
      // Extract smoothed RSSI
      const rssi1 =
        rssiHistory[device1.id]?.smoothedRssi ?? device1.rssi ?? -100;
      const rssi2 =
        rssiHistory[device2.id]?.smoothedRssi ?? device2.rssi ?? -100;

      // Convert RSSI -> distance
      const d1 = calculateDistance(rssi1, settings.A, settings.n);
      const d2 = calculateDistance(rssi2, settings.A, settings.n);

      // Compute the 1D parameter t along the corridor
      const tMeasured = computeT(d1, d2, corridorLength);

      // Kalman filter update
      kalmanFilterRef.current.processMeasurement(tMeasured);

      // Get the filtered t
      const smoothedT = kalmanFilterRef.current.currentT;

      // Convert t -> (x,y)
      const pos = getPositionOnCorridor(
        smoothedT,
        beaconPositions.beacon1,
        beaconPositions.beacon2
      );
      setUserPosition(pos);
    }
  };

  // If isCalculating, run calculatePosition every 0.1s
  useEffect(() => {
    let intervalId;
    if (isCalculating) {
      intervalId = setInterval(() => {
        calculatePosition();
      }, 100); // 10Hz
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isCalculating, devices, rssiHistory, settings, beaconPositions]);

  // Switch device selection
  const toggleDeviceSelection = (id) => {
    setSelectedDevices((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  // ***** Rendering List of Devices *****
  const renderDeviceItem = ({ item }) => {
    return (
      <View style={styles.deviceItem}>
        <Switch
          value={!!selectedDevices[item.id]}
          onValueChange={() => toggleDeviceSelection(item.id)}
        />
        <View style={{ marginLeft: 8 }}>
          <Text>Device ID: {item.id}</Text>
          <Text>Name: {item.name}</Text>
          <Text>RSSI: {item.rssi}</Text>
        </View>
      </View>
    );
  };

  // ***** Visualization with SVG *****
  const renderVisualization = () => (
    <Svg height="400" width="400" style={styles.svgContainer}>
      {/* Draw beacon1 */}
      <Circle
        cx={beaconPositions.beacon1.x}
        cy={beaconPositions.beacon1.y}
        r="10"
        fill="blue"
      />
      {/* Draw beacon2 */}
      <Circle
        cx={beaconPositions.beacon2.x}
        cy={beaconPositions.beacon2.y}
        r="10"
        fill="blue"
      />
      {/* User position */}
      <Circle
        cx={userPosition.x}
        cy={userPosition.y}
        r="10"
        fill="red"
      />
    </Svg>
  );

  // ***** Settings page for corridor and A/n *****
  const renderSettingsPage = () => (
    <View style={styles.settingsContainer}>
      <Text style={styles.settingsTitle}>Settings</Text>
      {/* Beacon1 coords */}
      <View style={styles.settingsRow}>
        <Text>Beacon1 X:</Text>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          onChangeText={(value) =>
            setBeaconPositions((prev) => ({
              ...prev,
              beacon1: {
                ...prev.beacon1,
                x: parseFloat(value) || 0,
              },
            }))
          }
          value={beaconPositions.beacon1.x.toString()}
        />
        <Text>Y:</Text>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          onChangeText={(value) =>
            setBeaconPositions((prev) => ({
              ...prev,
              beacon1: {
                ...prev.beacon1,
                y: parseFloat(value) || 0,
              },
            }))
          }
          value={beaconPositions.beacon1.y.toString()}
        />
      </View>

      {/* Beacon2 coords */}
      <View style={styles.settingsRow}>
        <Text>Beacon2 X:</Text>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          onChangeText={(value) =>
            setBeaconPositions((prev) => ({
              ...prev,
              beacon2: {
                ...prev.beacon2,
                x: parseFloat(value) || 0,
              },
            }))
          }
          value={beaconPositions.beacon2.x.toString()}
        />
        <Text>Y:</Text>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          onChangeText={(value) =>
            setBeaconPositions((prev) => ({
              ...prev,
              beacon2: {
                ...prev.beacon2,
                y: parseFloat(value) || 0,
              },
            }))
          }
          value={beaconPositions.beacon2.y.toString()}
        />
      </View>

      {/* A and n */}
      <View style={styles.settingsRow}>
        <Text>A:</Text>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          onChangeText={(value) =>
            setSettings((prev) => ({ ...prev, A: parseFloat(value) || -59 }))
          }
          value={settings.A.toString()}
        />
        <Text>n:</Text>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          onChangeText={(value) =>
            setSettings((prev) => ({ ...prev, n: parseFloat(value) || 2 }))
          }
          value={settings.n.toString()}
        />
      </View>
    </View>
  );

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <Text style={styles.title}>BLE Corridor Positioning (2 Beacons)</Text>

      {/* List of devices with selection */}
      <FlatList
        data={devices}
        keyExtractor={(item) => item.id}
        renderItem={renderDeviceItem}
      />

      <Button
        title={isCalculating ? 'Stop Positioning' : 'Start Positioning'}
        onPress={() => setIsCalculating(!isCalculating)}
      />

      {/* Visualization */}
      {renderVisualization()}

      {/* Settings */}
      {renderSettingsPage()}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  title: {
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
    marginVertical: 10,
  },
  deviceItem: {
    padding: 10,
    borderBottomWidth: 1,
    borderColor: '#ccc',
    flexDirection: 'row',
    alignItems: 'center',
  },
  svgContainer: {
    marginTop: 20,
    alignSelf: 'center',
  },
  settingsContainer: {
    padding: 20,
  },
  settingsTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
  },
  settingsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    flexWrap: 'wrap',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 5,
    marginHorizontal: 5,
    width: 50,
    textAlign: 'center',
  },
});

export default App;
