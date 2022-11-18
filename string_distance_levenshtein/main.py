from get_candidates import get_candidates

if __name__ == "__main__":
    print("kai: ", get_candidates("kai"))
    print("pelni: ", get_candidates("pelni"))
    print("banyumas: ", get_candidates("banyumas"))
    
    """
    Sample output:
    kai:  {'pt kai': 3, 'pt kereta api indonesia persero kai': 32}
    pelni:  {'pt pelni': 3, 'pt pelayaran nasional indonesia persero pelni': 40}
    banyumas:  {'pemkab banyumas': 7, 'pemerintah banyumas': 11, 'pemerintah kabupaten banyumas': 21}
    """
    