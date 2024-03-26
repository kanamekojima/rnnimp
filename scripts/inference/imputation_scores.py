import numpy as np


def calculate_INFO(genotype_probs, dosages=None, AF=None):
    if dosages is None:
        dosages = genotype_probs[:, 1] + 2 * genotype_probs[:, 2]
    if AF is None:
        AF = 0.5 * dosages.mean()
    if AF == 0 or AF == 1:
        return 1
    W = genotype_probs[:, 1] + 4 * genotype_probs[:, 2]
    return 1 - 0.5 * (W - np.square(dosages)).mean() / (AF * (1 - AF))


def calculate_Beagle_R2(genotype_probs, dosages=None, AF=None):
    genotypes = genotype_probs.argmax(axis=1)
    W = genotype_probs[:, 1] + 4 * genotype_probs[:, 2]
    if dosages is None:
        dosages = genotype_probs[:, 1] + 2 * genotype_probs[:, 2]
    if AF is None:
        AF = 0.5 * dosages.mean()
    numerator  = (genotypes * dosages).var()
    n = len(genotypes)
    denominator1 = W.sum() - np.square(dosages.sum()) / n
    denominator2 = np.square(genotypes).sum() - np.square(genotypes.sum()) / n
    return numerator / (denominator1 * denominator2)


def calculate_Minimac_R2(allele_probs, AF=None):
    if AF is None:
        AF = allele_probs.mean()
    if AF == 0 or AF == 1:
        return float('nan')
    return np.square(allele_probs - AF).mean() / (AF * (1 - AF))
