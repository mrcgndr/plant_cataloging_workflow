from cataloging.pipeline import Pipeline
import argparse

parser = argparse.ArgumentParser(description="Run the full plant cataloging pipeline with given config file.")
parser.add_argument("-c", dest="config_path", type=str, help="path to config file") 
args = parser.parse_args()

if __name__ == "__main__":
    
    # load pipeline with config file
    pipeline = Pipeline(args.config_path)
    
    # setup tasks and dependencies
    pipeline.build()
    
    # run the pipeline
    pipeline.run()
