
import os
import settings
import pandas as pd


HEADERS = {"data" : ["Creation_date",
                    "Status",
                    "Completion_date",
                    "Service_request_number",
                    "Type_service_request",
                    "If_yes_where_is_debris_located?",
                    "Current_activity",
                    "Most_recent_action",
                    "Street_address",
                    "Zip_code",
                    "X_coordinate",
                    "Y_coordinate",
                    "Ward",
                    "Police_district",
                    "Community_area",
                    "Lat",
                    "Lon",
                    "Location"]
          }

def concatenate():
  tree_cleaning_data.to_csv(os.path.join(settings.data, "{}.txt".format(prefix)), sep="|", header=SELECT[prefix], index=False)

if __name__ == "__main__":
    concatenate("data")
