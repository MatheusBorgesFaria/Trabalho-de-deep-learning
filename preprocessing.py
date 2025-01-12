import pandas as pd
import torch, os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class DataPreProcessing:
    
    DATASETS_FOLDER = os.getcwd() + "/datasets/backblaze/"
    SMART_IDX_TO_KEEP = [1, 5, 7, 184, 187, 188, 189, 190, 193, 194, 197, 198, 240, 241, 242]
    NUM_THREADS = 10
    SAVED_DATA_DIR = os.getcwd() + "/saved_data/"
    SAVED_DATA_PATH = SAVED_DATA_DIR + "tensor_data.pt"

    def __init__(self,):
        """
        Initializes the preprocessor with a list of file paths.
        :param dataset_folder: List of strings, paths to the CSV files.
        """
        
        # List of CSV file paths
        columns_to_keep = ["smart_"+str(idx)+"_normalized" for idx in self.SMART_IDX_TO_KEEP]
        
        self._dataset_folders = [folder for folder in os.listdir(self.DATASETS_FOLDER) if os.path.isdir(self.DATASETS_FOLDER + folder)]
        #print("self._dataset_folders = {}".format(self._dataset_folders))

        self._filenames = [self.DATASETS_FOLDER + folder + "/" + file for folder in self._dataset_folders for file in os.listdir(self.DATASETS_FOLDER + folder)]
        #print("self._filenames = {}".format(self._filenames))
        self._filenames = [(idx, filename) for idx, filename in enumerate(self._filenames)]
        #print("self._filenames = {}".format(self._filenames))

        if (os.path.isfile(self.SAVED_DATA_PATH)):
            print("#################################################################")
            print(f"Found saved data ! Loading tensor from {self.SAVED_DATA_PATH}...")
            print("#################################################################")
            self.data = torch.load(self.SAVED_DATA_PATH)
        else:
            print("#################################################################")
            print(f"No saved data found, loading CSV from {self.DATASETS_FOLDER}...")
            print("#################################################################")
            self.data = self.load_csvs_in_parallel(self._filenames, columns_to_keep, self.NUM_THREADS)

    def load_csv(self, file_path, columns_to_keep):
        """Loads a single CSV file and selects specific columns."""
        idx, file_path = file_path
        print("[{}]Reading {}...".format(idx, file_path))
        return pd.read_csv(file_path, usecols=columns_to_keep)

    def load_csvs_in_parallel(self, file_paths, columns_to_keep, num_threads=4):
        """Loads multiple CSV files in parallel using a thread pool."""
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks for parallel processing
            results = executor.map(lambda file: self.load_csv(file, columns_to_keep), file_paths)
        
        # Combine all dataframes
        print("Contatenating...")
        return pd.concat(results, ignore_index=True)

    def select_columns(self, columns=None, column_indexes=None):
        """
        Select specific columns to keep by name or index.
        :param columns: List of column names to keep.
        :param column_indexes: List of column indexes to keep.
        """
        if columns:
            self.data = self.data[columns]
        elif column_indexes:
            self.data = self.data.iloc[:, column_indexes]
        else:
            raise ValueError("You must specify either 'columns' or 'column_indexes'.")

    def remove_empty_or_useless_columns(self):
        """
        Removes columns that are entirely NaN or have only one unique value.
        """
        # Remove columns with all NaN values
        self.data.dropna(axis=1, how='all', inplace=True)

        # Remove columns with only one unique value (useless data)
        self.data = self.data.loc[:, self.data.nunique() > 1]

    def preview_data(self, num_rows=5):
        """
        Prints a preview of the data.
        :param num_rows: Number of rows to preview.
        """
        print(self.data.head(num_rows))

    def to_tensor(self):
        """
        Converts the preprocessed data to a PyTorch tensor.
        :return: PyTorch tensor.
        """
        return torch.tensor(self.data.values, dtype=torch.float32)

    def save_data(self):
        if not os.path.isdir(self.SAVED_DATA_DIR):
            os.mkdir(self.SAVED_DATA_DIR)
        torch.save(self.data, self.SAVED_DATA_PATH)


# Example Usage
if __name__ == "__main__":

    # Initialize preprocessor
    preprocessor = DataPreProcessing()

    # Preview initial data
    print("Initial Data Preview:")
    preprocessor.preview_data()
    
    # Select columns by name or index
    # preprocessor.select_columns(columns=columns_to_keep)

    # Remove empty or useless columns
    preprocessor.remove_empty_or_useless_columns()

    # Preview cleaned data
    print("Cleaned Data Preview (5 first items):")
    preprocessor.preview_data()

    # Convert to PyTorch tensor
    data_tensor = preprocessor.to_tensor()
    print("Data Tensor:", data_tensor)
    print("Data Tensor Shape:", data_tensor.shape)

    print(f"Saving data to {preprocessor.SAVED_DATA_PATH}...")
    preprocessor.save_data()
