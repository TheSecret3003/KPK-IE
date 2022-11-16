# Indexing

This folder contains codes to implement indexing and searching functionality.

## Reference Data
*preprocess_reference_data.py* module is used to preprocess the data to produce **reference_data**.
Reference data (stored in /data/indexing_v1/reference_data) consists of values from reference column in our labeled data (where status==yes) **and** the full *nama instansi* from KPK's lists.

## Index Table
*index.py* module is used to create indexing table based on previously created **reference_data**.
Index table (stored in /data/indexing_v1/reference_data) consists of 2 columns (Kata and Index). Index column contains the index or indexes in **reference_data** where the value of Kata exists.

## Search
*search.py* module is used to return entries in **reference_data**. The returned value will be the input to our similarity checking model(s).

### API
The search functionality is implemented as an API (REST).

- HTTP Method: GET
- Query String Param: nama_instansi
- URL: localhost:5000/get-references?nama_instansi=[nama_instansi here]

- To run the API, enter the following command:
    python app.py

To perform the API request:
1. Using CURL:
    - Enter the following in command line: curl "localhost:5000/get-references?nama_instansi=kementerian%20bumn"
2. Postman:
    - Create new GET request, enter the base url (localhost:5000/get-references), and add the query string parameter under *Params* menu
