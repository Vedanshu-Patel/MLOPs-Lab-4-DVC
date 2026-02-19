## Data Version Control (DVC)

### Project Description
This project demonstrates the use of DVC for data versioning in a machine learning pipeline. The ML task is **anomaly detection** on credit card customer data using PCA for dimensionality reduction and Isolation Forest to flag unusual spending behavior. The project tracks multiple versions of the dataset, processed data, models, and figures using DVC with Google Cloud Storage as the remote backend, enabling reproducibility and easy rollback to previous versions.

**Project Structure:**
```
DVC_LABS/
├── .dvc/               # DVC configuration
├── .github/workflows/  # GitHub Actions CI/CD
├── data/
│   ├── raw/
│   │   ├── CC_GENERAL.csv        # tracked by DVC
│   │   └── CC_GENERAL.csv.dvc    # DVC pointer file
│   └── processed/
│       ├── CC_PROCESSED.csv      # tracked by DVC
│       └── CC_PROCESSED.csv.dvc  # DVC pointer file
├── models/
│   ├── isolation_forest.joblib   # tracked by DVC
│   ├── pca.joblib                # tracked by DVC
│   └── scaler.joblib             # tracked by DVC
├── reports/
│   ├── figures/                  # tracked by DVC
│   │   ├── pca_variance.png      # PCA explained variance curve
│   │   ├── anomaly_scatter.png   # anomaly scatter plot
│   │   └── anomaly_scores.png    # anomaly score distribution
│   └── metrics.json              # tracked by Git
├── src/
│   ├── generate_data.py
│   ├── preprocess.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

- [DVC](https://dvc.org/) is an open-source tool that serves as a powerful asset in the machine learning project toolkit, with a primary focus on data versioning.
- **Data versioning** is a critical aspect of any ML project. It allows you to track changes and updates in your datasets over time, ensuring you can always recreate, compare, and reference specific dataset versions used in your experiments.
- In this lab tutorial, we will be utilizing DVC with Google Cloud Storage to enhance data versioning capabilities, ensuring efficient data management and collaboration within your machine learning project.

### Creating a Google Cloud Storage Bucket
1. Navigate to [Google Cloud Console](https://console.cloud.google.com/).
2. Ensure you've created a new project specifically for this lab.
3. In the Navigation menu, select "Cloud Storage," then go to "Buckets," and click on "Create a new bucket."
4. Assign a unique name to your bucket.
5. Select the region as `us-east1`
6. Proceed by clicking "Continue" until your new bucket is successfully created.
7. Once the bucket is created, we need to get the credentials to connect the GCP remote to the project. Go to the `IAM & Admin` service and go to `Service Accounts` in the left sidebar.
8. Click the `Create Service Account` button to create a new service account that you'll use to connect to the DVC project in a bit. Now you can add the name and ID for this service account and keep all the default settings. We've chosen `lab2` for the name. Click `Create and Continue` and it will show the permissions settings. Select `Owner` in the dropdown and click `Continue`.
9. Then add your user to have access to the service account and click `Done`. Finally, you'll be redirected to the `Service accounts` page. You'll see your service account and you'll be able to click on `Actions` and go to where you `Manage keys` for this service account.
10. Once you've been redirected, click the `Add Key` button and this will bring up the credentials you need to authenticate your GCP account with your project. Proceed by downloading the credentials in JSON format and securely store the file. This file will serve as the authentication mechanism for DVC when connecting to Google Cloud.

### Installing DVC with Google Cloud Support
- Ensure you have DVC with Google Cloud support installed on your system by using the following command:
	`pip install dvc[gs]`
- Note that, depending on your chosen [remote storage](https://dvc.org/doc/user-guide/data-management/remote-storage), you may need to install optional dependencies such as `[s3]`, `[azure]`, `[gdrive]`, `[gs]`, `[oss]`, `[ssh]`. To include all optional dependencies, use `[all]`.
- Run this command to setup google cloud bucket as your storage `dvc remote add -d myremote gs://<mybucket>`
- In order for DVC to be able to push and pull data from the remote, you need to have valid GCP credentials.
- Run the following command for authentication `dvc remote modify myremote credentialpath <YOUR JSON TOKEN LOCATION>`

### Tracking Data with DVC
- To initiate data tracking, execute the following steps:
	1. Run the `dvc init` command to initialize DVC for your project. This will generate a `.dvc` file that stores metadata and configuration details. Your `.dvc` file config metadata will look something like this:
	```
    [core]
        remote = myremote
    ['remote "myremote"']
        url = gs://<your-bucket-name>
        credentialpath = credentials.json
	```
	2. Generate the initial dataset (v1) and run the pipeline:
	```
	python src/generate_data.py --version 1
	python src/preprocess.py
	python src/train.py
	```
	3. Next, use `dvc add` to instruct DVC to start tracking all outputs:
	```
	dvc add data/raw/CC_GENERAL.csv
	dvc add data/processed/CC_PROCESSED.csv
	dvc add models/isolation_forest.joblib
	dvc add models/pca.joblib
	dvc add models/scaler.joblib
	dvc add reports/figures/pca_variance.png
	dvc add reports/figures/anomaly_scatter.png
	dvc add reports/figures/anomaly_scores.png
	```
	4. To ensure version control, add the generated `.dvc` pointer files to your Git repository:
	```
	git add data/raw/CC_GENERAL.csv.dvc
	git add data/processed/CC_PROCESSED.csv.dvc
	git add models/isolation_forest.joblib.dvc
	git add models/pca.joblib.dvc
	git add models/scaler.joblib.dvc
	git add reports/figures/pca_variance.png.dvc
	git add reports/figures/anomaly_scatter.png.dvc
	git add reports/figures/anomaly_scores.png.dvc
	git add reports/metrics.json
	```
	5. Also include the `.gitignore` files generated by DVC in each folder.
	6. Commit and push:
	```
	git commit -m "data: add all v1 outputs"
	dvc push
	git push
	```

- To push your data to the remote storage in Google Cloud, use the following DVC command: `dvc push`. This command will upload your data to the Google Cloud Storage bucket specified in your DVC configuration, making it accessible and versioned in the cloud.

### Handling Data Changes and Hash Updates
Whenever your dataset undergoes changes, DVC will automatically compute a new hash for the updated file. Here's how the process works:

- **Step 1 — Update the Dataset:** Replace the existing `CC_GENERAL.csv` with the updated version by running:
	```
	python src/generate_data.py --version 2
	```

- **Step 2 — Update DVC Tracking:** Re-add the file so DVC computes the new hash:
	```
	dvc add data/raw/CC_GENERAL.csv
	```

- **Step 3 — Rerun the Pipeline:** Regenerate processed data, models, and figures from the new dataset:
	```
	python src/preprocess.py
	python src/train.py
	```

- **Step 4 — Re-track Updated Outputs:** Update DVC tracking for all regenerated files:
	```
	dvc add data/processed/CC_PROCESSED.csv
	dvc add models/isolation_forest.joblib
	dvc add models/pca.joblib
	dvc add models/scaler.joblib
	dvc add reports/figures/pca_variance.png
	dvc add reports/figures/anomaly_scatter.png
	dvc add reports/figures/anomaly_scores.png
	```

- **Step 5 — Commit and Push:** Commit all updated `.dvc` pointer files to Git and push data to GCS:
	```
	git add data/raw/CC_GENERAL.csv.dvc
	git add data/processed/CC_PROCESSED.csv.dvc
	git add models/isolation_forest.joblib.dvc
	git add models/pca.joblib.dvc
	git add models/scaler.joblib.dvc
	git add reports/figures/pca_variance.png.dvc
	git add reports/figures/anomaly_scatter.png.dvc
	git add reports/figures/anomaly_scores.png.dvc
	git add reports/metrics.json
	git commit -m "data: update dataset to v2 and retrain model"
	dvc push
	git push
	```

#### Reverting to Previous Versions with Hashes

To revert to a previous version (e.g., dataset v1):

- **Step 1 — Check commit history:**
	```
	git log --oneline
	```
	You will see something like:
	```
	efbde92 feat: retrain model on dataset v2
	5188e66 data: update CC_GENERAL.csv v2 (9500 rows)
	70e20ed feat: run ML pipeline on dataset v1
	e1ba699 data: v1 (8950 rows)
	```

- **Step 2 — Checkout the v1 commit:**
	```
	git checkout e1ba699
	```

- **Step 3 — Restore the v1 data using DVC:**
	```
	dvc checkout
	```
	DVC uses the hash stored in the `.dvc` pointer files to fetch the correct version from GCS.

- **Step 4 — Rerun the pipeline on v1 data:**
	```
	python src/preprocess.py
	python src/train.py
	```
	Check `reports/metrics.json` — `n_samples` should show `8950` confirming you are on v1.

- **Step 5 — Return to latest version (v2):**
	```
	git checkout master
	dvc checkout
	python src/preprocess.py
	python src/train.py
	```
	Check `reports/metrics.json` — `n_samples` should show `9500` confirming you are back on v2.

> 💡You can follow [this](https://www.youtube.com/watch?v=kLKBcPonMYw&list=PL7WG7YrwYcnDb0qdPl9-KEStsL-3oaEjg&pp=iAQB) tutorial to learn about DVC in detail.