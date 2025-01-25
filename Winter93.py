import numpy as np
np.random.seed(1)


def quantity_it(A_it,K_it):
    return A_it*K_it

def price_t(Q_t, D, eta):
    return min(D/Q_t**(1/eta),1.20)

def profit_rate(A_it, p_t, r_im, r_in, c):
    return p_t*A_it - c - r_im - r_in

def dummy_imt(r_im, K_it, a_m):
    prob_imt = a_m*r_im*K_it
    prob_imt = min(max(prob_imt, 0),1)
    return np.random.choice([0, 1], p=[1 - prob_imt, prob_imt])
    
def dummy_int(r_in, K_it, a_n):
    prob_int = a_n*r_in*K_it
    prob_int = min(max(prob_int, 0),1)
    return np.random.choice([0, 1], p=[1 - prob_int, prob_int])

# Entrepreneurship Regime
def innovER(mu, sigma , t):
    lambda_t = mu + 0.01*t
    return np.random.normal(loc=lambda_t, scale=sigma)

#Routinized Regime
def innovRR(mu, sigma, t, A_it):
    lambda_t = .5*(mu + np.log(A_it)) + 0.01*t
    return np.random.normal(loc=lambda_t, scale=sigma/5)

#Para la nueva empresa entrante hay un nivel básico de conocimiento público. Por lo tanto, eso es el nivel de conocimiento con el que realizan la nueva innov.

def A_it1(A_it, d_int, d_imt, t, Amax, mu, sigma, RR):
    patent = 0
    if RR == False:
        innov_it = np.exp(innovER(mu, sigma, t)) 
    else:
        innov_it = np.exp(innovRR(mu, sigma, t, A_it))    
    tildeA_it = A_it + d_int*(innov_it - A_it)
    hatA_it = A_it + d_imt*(Amax - A_it)

    if max(A_it, hatA_it, tildeA_it) == innov_it:
        patent = 1
        
    return max(A_it, hatA_it, tildeA_it), patent

def investment(p_t, A_it, Q_t, q_it, pi_it, delta, c, eta_d, eta_s, BANK):
    if pi_it<=0:
        fpi = delta + pi_it
    else:
        fpi = delta + (1 + BANK)*pi_it
    rho = p_t*A_it/float(c)
    s = q_it/float(Q_t)
    if s < 1 - 1e-8 :
        mu = float(eta_d + (1-s)*eta_s)/float(eta_d + (1-s)*eta_s - s)
        I_d = 1 + delta - 1/rho*mu
    else:
        I_d = 1 - c/p_t
    return max(0,min(I_d,fpi))

def capitaldynamic(K_it, I_it, delta):
    return (1 - delta + I_it)*K_it

def adaptativeRD(X_it, pi_t, r_mt, r_nt, r_imt, r_int, beta):
    if X_it < pi_t and np.random.choice([0, 1]) == 1:
        sigma_m = 0.0004
        sigma_n = 0.002
        u_int = abs(np.random.normal(0, sigma_n))
        u_imt = abs(np.random.normal(0, sigma_m))
        return (1 - beta)*r_imt + beta*r_mt + u_imt, (1 - beta)*r_int + beta*r_nt + u_int
    else:
        return r_imt, r_int

def performance(X_it, pi_it, theta):
    return theta*X_it + (1-theta)*pi_it


def entry(a_m, E_m, a_n, E_n):
    #Número de potenciales entrantes en el periodo t, imitadores-innovadores
    M_t = a_m*E_m
    N_t = a_n*E_n
    imitators = np.random.poisson(M_t)
    innovators = np.random.poisson(N_t)
    return imitators, innovators

def entrycondition(p_t, A_it, r_e, c):
    sigma_et = 0.014
    u_et = np.random.normal(0, sigma_et)
    if p_t*A_it - c> r_e + u_et:
        return 1

def media(lista, condiciones):
    filtrados = [x for x, *rest in zip(lista, *condiciones) if all(rest)]
    return sum(filtrados) / len(filtrados) if filtrados else 0

def NWCap12(t = 100, n = 32, prop = 0.5, patentime = 5, RR = False):

    eta = 1
    D = 64
    c = .16
    a_m = 2.5
    a_n = .25
    delta = 0.03
    K0 = 139.58
    A0 = 0.16
    mu = np.log(0.135)
    sigma = 0.1177
    X0 = 0.001
    Kmin = 10
    Xmin = -0.051
    XDelta = 0.001
    beta = 0.167
    theta = 0.75
    r_imt0 = 0.002
    r_int0 = 0.005
    E_n = .2
    E_m = 0.2
    r_e = 0.007
    eta_s = 2
    eta_d = 1
    BANK = 1

    
    r_imt, r_int, A_it, X_it, K_it = [[r_imt0]*n], [[r_int0]*n], [[A0]*n], [[X0]*n], [[K0]*n]
    dimt_it, dint_it, pi_it, q_it, I_it, patent_it = [], [], [], [], [], []
    p_t, pi_t, rm_t, rn_t, Q_t = [], [], [], [], []

    #ID indica si la firma es innovadora si es imitadora
    ID = [1]*round(n*prop) + [0]*(n-round(n*prop))

    Viva = [0]*n

    
    for t in range(0, t + 1):
        n = len(ID)
        
        if t >= patentime:
            Amax = max(A_it[t - patentime])
        else:
            Amax = 0.16
            
        # We extend every list to incorporate the new values.
        r_imt.append([0]*n), r_int.append([0]*n), A_it.append([0]*n), X_it.append([0]*n), K_it.append([0]*n)
        dimt_it.append([0]*n), dint_it.append([0]*n), pi_it.append([0]*n), q_it.append([0]*n), I_it.append([0]*n), patent_it.append([0]*n)

        for i in range(0,n):
            # If they are imitators, then they dont use resources for innovation.
            if ID[i] == 0:
                r_int[t][i] = 0

            #If they are alive then we calculate quantities
            if Viva[i] >= 0:
                q_it[t][i] = quantity_it(A_it[t][i],K_it[t][i])

        #We determine the quantities and the price
        Q_t.extend([sum(q_it[t])])
        if Q_t[-1] == 0: 
            break
        p_t.extend([price_t(Q_t[t], D, eta)])
        
        for i in range(0,n):
            if Viva[i] >= 0:
                # Before updating the next technology we have to establish if its gamble was succesful or if it wasn't.
                dimt_it[t][i] = dummy_imt(r_imt[t][i], K_it[t][i], a_m)
                if ID[i] == 1:
                    dint_it[t][i] = dummy_int(r_int[t][i], K_it[t][i], a_n)
                A_it[t+1][i], patent_it[t][i] = A_it1(A_it[t][i], dimt_it[t][i], dint_it[t][i], t, Amax, mu, sigma, RR)
                
                # Calculate profit and Investment. 
                pi_it[t][i] = profit_rate(A_it[t][i], p_t[t], r_imt[t][i], r_int[t][i], c)
                I_it[t][i] = investment(p_t[t], A_it[t+1][i], Q_t[t], q_it[t][i], pi_it[t][i], delta, c, eta_d, eta_s, BANK)

                # We update the values for next period
                K_it[t+1][i] = capitaldynamic(K_it[t][i], I_it[t][i], delta)
                X_it[t+1][i] = performance(X_it[t][i], pi_it[t][i], theta)



        # Only if they were alive for that period, we calculate their means.
        pi_t.append(media(pi_it[t], [Viva]))
        rn_t.append(media(r_int[t], [Viva, [i == 1 for i in ID]]))
        rm_t.append(media(r_imt[t], [Viva]))

        
        # After they know how they performed, we determine if they still alive for next period and if they adapt their research and development policies.
        for i in range(0,n):
            if Viva[i] >= 0:
                if K_it[t+1][i] < Kmin or X_it[t+1][i] < Xmin:
                    Viva[i] = -t # Si no cumple standards muere.
                
            if Viva[i] >= 0:
                r_imt[t+1][i], r_int[t+1][i] = adaptativeRD(X_it[t][i], pi_t[t], rm_t[t], rn_t[t], r_imt[t][i], r_int[t][i], beta)
                
                #Si actualiza, entonces:
                if r_int[t+1][i] != r_int[t][i] or r_imt[t+1][i] != r_imt[t][i]: 
                    X_it[t+1][i] += XDelta 


        # Entry New Firms Block
        imitators, innovators = entry(a_m, E_m, a_n, E_n)
        for i in range(imitators):
            if entrycondition(p_t[t], Amax, r_e,c) == 1:
                ID.extend([0])
                Viva.extend([t+1])
                K0 = max(np.random.normal(25,7.5),Kmin)
                A_it[t+1].extend([Amax])
                K_it[t+1].extend([K0])
                X_it[t+1].extend([X0])
                r_imt[t+1].extend([r_imt0])
                r_int[t+1].extend([0])
        
        for i in range(innovators):
            if RR == False: 
                A0 = np.exp(innovER(mu,sigma,t))
            else:
                A0 = np.exp(innovRR(mu,sigma,t,.1333))
            if entrycondition(p_t[t], A0, r_e,c) == 1:
                ID.extend([1])
                Viva.extend([t+1])
                K0 = max(np.random.normal(25,7.5),Kmin)
                patent_it[t].extend([1])
                A_it[t+1].extend([A0])
                K_it[t+1].extend([K0])
                X_it[t+1].extend([X0])
                r_imt[t+1].extend([r_imt0])
                r_int[t+1].extend([r_int0])
        
    return NWCap12Results(r_imt, r_int, A_it, X_it, K_it, dimt_it, dint_it, pi_it, q_it, I_it, p_t, pi_t, rm_t, rn_t, Q_t, ID, Viva, patent_it)

class NWCap12Results:
    def __init__(self, r_imt, r_int, A_it, X_it, K_it, dimt_it, dint_it, pi_it, q_it, I_it, p_t, pi_t, rm_t, rn_t, Q_t, ID, Viva, patent_it):
        self.r_imt = r_imt
        self.r_int = r_int
        self.A_it = A_it
        self.X_it = X_it
        self.K_it = K_it
        self.dimt_it = dimt_it
        self.dint_it = dint_it
        self.pi_it = pi_it
        self.q_it = q_it
        self.I_it = I_it
        self.p_t = p_t
        self.pi_t = pi_t
        self.rm_t = rm_t
        self.rn_t = rn_t
        self.Q_t = Q_t
        self.ID = ID
        self.Viva = Viva
        self.patent_it = patent_it