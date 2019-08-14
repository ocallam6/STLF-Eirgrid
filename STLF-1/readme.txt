================================================================
SHORT TERM LOAD FORECASTING ROI
================================================================
--------Changes to be made for REAL FORECAST--------------------
		
		Every day the network will need to be trained on a new file for forecasting a clone of that to local file would be useful rather than manual copy and paste every day.

		Currently only works for day ahead exactly: from 00:00 to 00:00. Read in algorithm doesn't cope with other day aheads, i.e from 13:20 to 13:20.

1. Raw_To_CSV.py:
	Takes in raw load and weather data.
	Converts to usable CSV file.

2. cleancsv.py:
	Uses the CSV file created previously and then cleans it by linearly interpolating and repeated or unexpected zero values. Updates CSV file and saves.

3. readfile.py:
	This find the dates to be exlcuded in the raw data and makes it usable for data.py.

4. data.py:
	This creates a timestamped dataframe from the clean CSV file containing relevant data (load, weather etc.) for the forecast.

5. traintest.py:
	One function splits the dataframe returned by data.py into a train, test split where the test is just the LAST DAY.
	The other function adds 48 rows with zero load value and the predicted tempmax for the next day.

6. network.py:
	This defines the neural network. See model.png for an overview. Uses Keras. Parameters may need tuning as time goes on. Regression based neural network.

7. analysis.py:
	One function graphs the predicted values.
	Another gives the statistical summary.
	Another returns predicted values for the next day.

8. main.py:
	Takes an input of two dates and then calls previous files.
	Epochs defines here as the number of times data passes through. Set epochs=0 to use the configuration set in weights.best.hdf5 .
