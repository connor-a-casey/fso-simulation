%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMULATION DESCRIPTION
% This script performs Line-of-Sight (LOS) analysis for:
% - 1 satellite
% - 3 ground stations
%
% Output files:
% 1. AccessIntervals.xlsx: This file contains the duration (in seconds) of 
%    each access interval between the satellite and the ground stations.
%    Columns: {'Source', 'Target', 'IntervalNumber', 'StartTime', 'EndTime', 
%    'Duration', 'StartOrbit', 'EndOrbit'}
% 
% 2. LookAngles.xlsx: This file contains the elevation and azimuth angles
%    for each time interval during the access periods, with a sample time 
%    of 1 minute.
%    Columns: {'Source', 'Target', 'IntervalNumber', 'Time', 'Elevation', 
%    'Azimuth', 'StartOrbit', 'EndOrbit'}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize scenario
startTime = datetime(2020,5,1,11,36,0);
stopTime = startTime + days(1);
sampleTime = 60; % 60 seconds
sc = satelliteScenario(startTime,stopTime,sampleTime);

% Read TLE file and add satellite
tleFile = 'ceb.tle'; %The TLE file must contain information about just one satellite
sat = satellite(sc, tleFile);
disp(['Successfully added satellite from file: ', tleFile]);

% Define ground stations
latitudes = [10, 20, 30];
longitudes = [-30, -40, -50];

% Create ground stations
groundStations = arrayfun(@(lat, lon) groundStation(sc, lat, lon), latitudes, longitudes);

% Initialize an empty table to store accesses, azimuths and elevations based on time intervals
allIntervals = table();
allLookAngles = table();

% Access intervals for the satellite and each ground station
for j = 1:length(groundStations)
    ac = access(sat, groundStations(j));
    intvls = accessIntervals(ac);
    intvls.Source = repmat({char(sat.Name)}, height(intvls), 1);
    intvls.Target = repmat({['Ground station ', num2str(j)]}, height(intvls), 1);
    
    % Initialize arrays to store elevation, azimuth, and time data
    elevation = [];
    azimuth = [];
    startTimes = [];
    intervalNumber = [];
    startOrbit = [];
    endOrbit = [];
    
    for k = 1:height(intvls)
        times = intvls.StartTime(k):seconds(sampleTime):intvls.EndTime(k);
        
        % Initialize arrays to store elevation and azimuth for each time step
        el = [];
        az = [];
        
        for t = 1:length(times)
            % Calculate the satellite positions in ECEF coordinates
            satPos = transpose(states(sat, times(t), 'CoordinateFrame', 'ecef'));
            gsPos = [groundStations(j).Latitude, groundStations(j).Longitude, groundStations(j).Altitude];
            
            % Calculate elevation and azimuth
            [azimuthValue, elevationValue] = lookangles(gsPos, satPos);
            el = [el; elevationValue];
            az = [az; azimuthValue];
        end
        
        elevation = [elevation; el];
        azimuth = [azimuth; az];
        startTimes = [startTimes; times'];
        intervalNumber = [intervalNumber; repmat(intvls.IntervalNumber(k), length(times), 1)];
        startOrbit = [startOrbit; repmat(intvls.StartOrbit(k), length(times), 1)];
        endOrbit = [endOrbit; repmat(intvls.EndOrbit(k), length(times), 1)];
    end
    
    % Create a table to add data of interest
    lookAnglesTable = table(repmat({char(sat.Name)}, length(startTimes), 1), ...
                            repmat({['Ground station ', num2str(j)]}, length(startTimes), 1), ...
                            intervalNumber, startTimes, elevation, azimuth, startOrbit, endOrbit, ...
                            'VariableNames', {'Source', 'Target', 'IntervalNumber', 'Time', 'Elevation', 'Azimuth', 'StartOrbit', 'EndOrbit'});
    
    % Combine the intervals into the main table
    allIntervals = [allIntervals; intvls];
    allLookAngles = [allLookAngles; lookAnglesTable];
    
    disp(['Access intervals for Satellite and Ground Station ', num2str(j)]);
    disp(intvls);
end

% Write the combined tables to Excel files
writetable(allIntervals, 'AccessIntervals.xlsx');
writetable(allLookAngles, 'TimeStamp.xlsx');

% Play the scenario to visualize the satellite
play(sc);
