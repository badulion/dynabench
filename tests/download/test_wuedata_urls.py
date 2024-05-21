from dynabench.dataset._download import DATASETS_WUEDATA, BASE_URL
import requests

def test_wuedata_urls():
    for equation in ["burgers", "gasdynamics", "kuramotosivashinsky", "reactiondiffusion", "wave"]:
        for structure in ["cloud", "grid"]:
            for resolution in ["full", "high", "low", "medium"]:
                if structure == "cloud" and resolution == "full":
                    continue
                key = f"{equation}-{structure}-{resolution}"
                assert key in DATASETS_WUEDATA
                assert BASE_URL % DATASETS_WUEDATA[key] == f"https://wuedata.uni-wuerzburg.de/radar/api/datasets/{DATASETS_WUEDATA[key]}/download"

                url = f"https://wuedata.uni-wuerzburg.de/radar/api/datasets/{DATASETS_WUEDATA[key]}/metadata"
                response = requests.get(url)
                assert response.status_code == 200


test_wuedata_urls()