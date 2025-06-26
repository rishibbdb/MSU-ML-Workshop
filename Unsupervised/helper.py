import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sindec_bandwidth=np.radians(2)
solid_angle = 2 * np.pi * sindec_bandwidth

def generate_random_data(n_samples, random_state=0):
    np.random.seed(random_state)
    X = np.random.uniform(0,10,(n_samples, 2))
    return X

def generate_random_dist(start, stop, n_samples, random_state=0):
    np.random.seed(random_state)
    large_sample = np.random.uniform(start, stop, (n_samples, 2))
    return large_sample 


def dbscan_clustering(X, eps=0.1, min_samples=5):
    X_scaled = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_
    return labels

def apply_gmm(X, n_components=3):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    log_likelihood = gmm.score(X)
    return labels, log_likelihood, gmm

def visualize_clustering(X, labels, title):
    plt.figure(figsize=(10,8))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    
    plt.title(title)
    plt.show()

def tune_dbscan(X):
    best_eps = 0.2
    best_min_samples = 20
    best_score = -2 
    
    eps_range = np.arange(0.15, 0.5, 0.05)
    min_samples_range = np.arange(20, 100, 1)
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            if len(set(labels)) > 1 and -1 not in labels:
                sscore = silhouette_score(X, labels)
                num_noise = np.sum(labels == -1)
                noise_loss = num_noise / len(labels) 
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                cluster_loss = abs(num_clusters - 3)  # Penalize if clusters are too many/few (3 is arbitrary)
                score = sscore + weight_noise * noise_loss + weight_clusters * cluster_loss
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
    
    print(f"Best DBSCAN params: eps={best_eps}, min_samples={best_min_samples}, silhouette score={best_score:.2f}")
    return best_eps, best_min_samples
    
#############Injection data##################
def map_to_latlonz(m, N=1000):
    x = np.linspace(np.pi, -np.pi, N)
    y = np.linspace(np.pi, 0, N)
    X, Y = np.meshgrid(x, y)

    r = healpy.rotator.Rotator(rot=(-180, 0, 0))
    YY, XX = r(Y.ravel(), X.ravel())
    pix = healpy.ang2pix(healpy.get_nside(m), YY, XX)
    Z = np.reshape(m[pix], X.shape)

    lon = x[::-1]
    lat = np.pi/2 - y
    return lat, lon, Z

def set_indices(sig, src, sindec_bandwidth):

    sig, src, sindec_bandwidth = sig, src, sindec_bandwidth
    src_sindecs = np.sin(src.dec)
    min_sindecs = np.maximum(-1, src_sindecs - .5*sindec_bandwidth)
    max_sindecs = np.minimum(+1, src_sindecs + .5*sindec_bandwidth)
    min_sindecs[max_sindecs == +1] = +1 - sindec_bandwidth
    max_sindecs[min_sindecs == -1] = -1 + sindec_bandwidth
    min_decs, max_decs = np.arcsin(min_sindecs), np.arcsin(max_sindecs)

    indices = []
    N = len(src)
    logging = N > 50
    flux0 = self.flux[0]
    energy_min, energy_max = flux0.energy_range
    mask_energy = ((energy_min <= sig.true_energy)
                    & (sig.true_energy <= energy_max))
    flush = sys.stdout.flush

    for (i, (min_dec, max_dec, flux)) in enumerate(izip(min_decs, max_decs, self.flux)):
        if flux.energy_range != (energy_min, energy_max):
            energy_min, energy_max = flux.energy_range
            this_mask_energy = ((energy_min <= sig.true_energy)
                                & (sig.true_energy <= energy_max))
        else:
            this_mask_energy = mask_energy
        mask_dec = (min_dec <= sig.true_dec) & (sig.true_dec <= max_dec)
        mask = mask_dec & this_mask_energy
        indices.append(np.where(mask)[0])
    return indices


def set_weights(sig, src, indices, flux):
    weights = [1]
    sig, src = sig, src
    ow, E = ana.livetime / solid_angle * sig.oneweight, sig.true_energy
    n_src = len(src)
    for i_src in xrange(n_src):
        indices = indices[i_src]
        weight = ow[indices] * flux[i_src](E[indices])
        weights.append(weight)
    weights = weights   # Can be removed to reduce memory consumption?
    src_weights = src.weight * np.array([np.sum(weight) for weight in weights])
    probs = [weights[i] / np.sum(weights[i]) for i in xrange(n_src)]
    src_prob = src_weights / np.sum(src_weights)
    acc_total = np.sum(src_weights)

def PowerLawFlux(gamma, norm=1, energy_range=(0, np.inf), energy_cutoff=np.inf):

    self.gamma, self.norm = gamma, norm
    self._energy_range, self.energy_cutoff = tuple(energy_range), energy_cutoff


def inject(n_inj, seed=None):
    random = get_random(seed)
    src_n_injs = random.multinomial(n_inj, self.src_prob)
    sig, src = self.sig, self.src
    n_src = len(self.src)
    outs = []
    for i_src in xrange(n_src):
        src_n_inj = src_n_injs[i_src]
        if src_n_inj == 0:
            continue
        # collect source info
        indices = self.indices[i_src]
        probs = self.probs[i_src]
        extension = src.extension[i_src]
        src_ra, src_dec = src.ra[i_src], src.dec[i_src]
        # draw events
        idx = random.choice(indices, src_n_inj, p=probs)
        ev_ra, ev_dec = sig.xra[idx], sig.xdec[idx]
        # smear if necessary
        if extension:
            ev_ra += random.normal(0, extension, src_n_inj)
            ev_dec += random.normal(0, extension, src_n_inj)
        # rotate to source position
        ev_dec, ev_ra = coord.rotate_xaxis_to_source(
            src_dec, src_ra, ev_dec, ev_ra, latlon=True)
        out = sig[idx]
        out['ra'], out['dec'] = ev_ra, ev_dec
        out['idx'] = idx
        out['inj'] = np.repeat(self, len(out))
        for key in out.keys():
            if key not in self.keep:
                del out[key]
        #out.compress(self.keep + 'idx inj'.split())
        outs.append(out)
    return outs, 0



def plot_gaussian_clusters(data, labels, cluster_params):
    fig, ax = plt.subplots()

    # Scatter plot of the data points
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)

    # Plot Gaussian ellipses for each cluster
    for mean, cov, _ in cluster_params:
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals)
        ellipse = plt.matplotlib.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='r', facecolor='none')
        ax.add_patch(ellipse)
    plt.scatter(mean[0], mean[1], color='red', marker='x', s=100, label='Minuit Min')
    plt.scatter(sig_ra_rad2, sig_dec_rad2, s=0.2, label='Signal Events')
    ax.set_title('DBSCAN Clustering with Gaussian Fits (Minuit)')
    plt.colorbar(scatter)
    plt.show()


def gauss_negative_log_likelihood(mu_x, mu_y, sigma_x, sigma_y, rho, points):
    mean = np.array([mu_x, mu_y])
    cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                    [rho*sigma_x*sigma_y, sigma_y**2]])

    if not np.all(np.linalg.eigvals(cov) > 0):
        return np.inf  

    try:
        mvn = multivariate_normal(mean=mean, cov=cov)
        log_likelihood = np.sum(np.log(np.maximum(mvn.pdf(points), 1e-9)))
    except np.linalg.LinAlgError:
        return np.inf  # Handle errors if covariance matrix is singular

    return -log_likelihood

def fit_gaussian_minuit(cluster_points):
    mu_x_init, mu_y_init = np.mean(cluster_points, axis=0)
    sigma_x_init, sigma_y_init = np.std(cluster_points, axis=0)
    rho_init = 0 

    def nll_helper(mu_x, mu_y, sigma_x, sigma_y, rho):
        return negative_log_likelihood(mu_x, mu_y, sigma_x, sigma_y, rho, cluster_points)

    m = Minuit(nll_helper, mu_x=mu_x_init, mu_y=mu_y_init,
               sigma_x=sigma_x_init, sigma_y=sigma_y_init, rho=rho_init)

    # parameter limits 
    m.limits['sigma_x'] = (1e-2, 1)
    m.limits['sigma_y'] = (1e-2, 1)
    m.limits['rho'] = (-1, 1)

    m.migrad()

    mu_x, mu_y = m.values['mu_x'], m.values['mu_y']
    sigma_x, sigma_y = m.values['sigma_x'], m.values['sigma_y']
    rho = m.values['rho']

    mean = np.array([mu_x, mu_y])
    cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                    [rho*sigma_x*sigma_y, sigma_y**2]])

    return mean, cov, m.fval 

def make_cluster_plots(data, labels, cluster_params, make_all=False):

    fig = plt.figure(figsize=(16, 15))
    ax1 = fig.add_subplot(311)
    scatter = ax1.scatter(data[:, 0], data[:, 1], s=1, c=labels, cmap='seismic', alpha=0.6)
    injected_events = ax1.scatter(sig_ra_rad2, sig_dec_rad2, s=1, label='Signal Events', color='green')

    if make_all:
        ax2 = fig.add_subplot(312)
        background = ax2.scatter(bkg_ra_rad2, bkg_dec_rad2, s=1, label='Background Events')
        signal = ax2.scatter(sig_ra_rad2, sig_dec_rad2, s=1, label='Signal Events')
        ax2.scatter(np.pi-4.64,-0.505, s=40, color='red', label='Galactic Center')
        ax2.legend(loc='lower left')

        ax3 = fig.add_subplot(313)
        signal = ax3.scatter(sig_ra_rad2, sig_dec_rad2, s=1, label='Signal Events')
        ax3.scatter(np.pi-4.64,-0.505, s=40, color='red', label='Galactic Center')
        ax3.legend(loc='lower left')

        ax2.set_xlim(-1.75, -1.25)
        ax2.set_ylim(-0.7, -0.3)
        ax3.set_xlim(-1.75, -1.25)
        ax3.set_ylim(-0.7, -0.3)

    for mean, cov, _, label in cluster_params:
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals)
        ellipse = plt.matplotlib.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='r', facecolor='none')
        ellipse2 = plt.matplotlib.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='r', facecolor='none')
        ax1.add_patch(ellipse)
        ax1.scatter(mean[0], mean[1], color='red', marker='x', s=100, label=f'Cluster {label} Centroid')
        if make_all:
             ax3.add_patch(ellipse2)
    ax1.set_xlim(-1.75, -1.25)
    ax1.set_ylim(-0.7, -0.3)

    plt.legend(loc='lower left')
    # plt.colorbar(scatter)
    plt.show()