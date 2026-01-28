import numpy as np
import os

# code to generate the NACA-4 airfoil coordinates

def naca4(m, p, t, n=200):
    x = np.linspace(0, 1, n)
    yt = 5 * t * (
        0.2969*np.sqrt(x)
        - 0.1260*x
        - 0.3516*x**2
        + 0.2843*x**3
        - 0.1015*x**4
    )

    yc = np.where(
        x < p,
        m/p**2 * (2*p*x - x**2),
        m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2)
    )

    dyc_dx = np.where(
        x < p,
        2*m/p**2*(p-x),
        2*m/(1-p)**2*(p-x)
    )

    theta = np.arctan(dyc_dx)

    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)

    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])

    return np.stack([x_coords, y_coords], axis=1)


#plot the airfoil cordninates
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    out_dir = "./airfoils_npy"
    os.makedirs(out_dir, exist_ok=True)

    t = 0.12
    for m_digit in range(1, 9):      # 0..8 -> leading digit
        for p_digit in range(1, 9):  # 0..4 -> second digit
            m = m_digit / 100.0
            p = p_digit / 10.0
            coords = naca4(m, p, t)
            code = f"{m_digit}{p_digit}12"
            fname = os.path.join(out_dir, f"{code}.npy")
            np.save(fname, coords)
            print(f"Saved {fname}")
    


