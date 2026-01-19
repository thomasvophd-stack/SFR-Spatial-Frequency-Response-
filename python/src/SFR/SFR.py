import json
import numpy as np

class SFR:
    def __init__(self, image_path, config_path):
        """
        Initialize the SFR class with star formation rate data.

        Parameters:
        sfr_data (list of tuples): Each tuple contains (time, sfr) values.
        """
        self.times, self.sfr_values = zip(*sfr_data)
        self.times = np.array(self.times)
        self.sfr_values = np.array(self.sfr_values)

        sfr_config = self.load_config(config_path)

        self.pixel_size = sfr_config.get('pixel_size', 1.0)
    
    @staticmethod
    def load_config(config_path):
        """
        Load the SFR configuration from a JSON file.

        Parameters:
        config_path (str): Path to the JSON configuration file.

        Returns:
        dict: Configuration parameters for SFR.
        """
        try:
            # Open the file in read mode ('r')
            with open(config_path, 'r', encoding='utf-8') as file:
                # Load the JSON data from the file
                config = json.load(file)
                
        except FileNotFoundError:
                print("Error: The file 'data.json' was not found.")
        except json.JSONDecodeError as e:
                print(f"Error: Failed to decode JSON from the file: {e}")

        return config

    def get_sfr_at_time(self, time):
        """
        Get the star formation rate at a specific time using linear interpolation.

        Parameters:
        time (float): The time at which to get the SFR.

        Returns:
        float: The interpolated SFR value at the given time.
        """
        return np.interp(time, self.times, self.sfr_values)

    def to_json(self):
        """
        Serialize the SFR data to a JSON string.

        Returns:
        str: JSON representation of the SFR data.
        """
        sfr_data = [{'time': t, 'sfr': s} for t, s in zip(self.times, self.sfr_values)]
        return json.dumps(sfr_data)

    @classmethod
    def from_json(cls, json_str):
        """
        Deserialize the SFR data from a JSON string.

        Parameters:
        json_str (str): JSON representation of the SFR data.

        Returns:
        SFR: An instance of the SFR class.
        """
        sfr_data = json.loads(json_str)
        data_tuples = [(item['time'], item['sfr']) for item in sfr_data]
        return cls(data_tuples)