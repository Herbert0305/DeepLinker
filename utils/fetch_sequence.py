import requests as r
baseUrl = "http://www.uniprot.org/uniprot/"
def fetch_uniprot_sequence(uniprot_id):
    try:
        currentUrl = baseUrl + uniprot_id + ".fasta"
        print(f"Requesting target sequence from: {currentUrl}")
        response = r.get(currentUrl)
        print(response.status_code)
        cData = ''.join(response.text)
        i = cData.index('\n') + 1
        temp_seq = cData[i:].strip()
        seq = temp_seq.replace('\n', '') 
        return seq
    except Exception as e:
        print(f"Error fetching target sequence: {e}")
        return ""
    