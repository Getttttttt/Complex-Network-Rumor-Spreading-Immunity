import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import EoN

def load_network(file_path):
    # Load the edge list from the Excel file
    edge_list_path = file_path
    edge_list_df = pd.read_excel(edge_list_path)

    # Create a graph from the edge list
    # G = nx.from_pandas_edgelist(edge_list_df, source='source', target='target')
    
    G = nx.from_pandas_edgelist(edge_list_df, source='source', target='target')

    # Display basic information about the network to confirm successful loading
    network_info = {
        "Number of Nodes": G.number_of_nodes(),
        "Number of Edges": G.number_of_edges(),
        "Is the Network Directed": nx.is_directed(G)
    }
    
    print(network_info)
    
    return G

def rich_club(G):
    # Calculate the rich-club coefficient for the network
    rich_club_dict_small = nx.rich_club_coefficient(G, normalized=True, Q=100)
    degrees_small, coefficients_small = zip(*rich_club_dict_small.items())

    # Plot the rich-club coefficient for the smaller network
    plt.figure(figsize=(10, 6))
    plt.plot(degrees_small, coefficients_small, 'b-', lw=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('Rich-Club Coefficient', fontsize=14)
    plt.title('Rich-Club Coefficient vs Degree', fontsize=16)
    plt.grid(True, which="both", ls="--")
    plt.savefig('./Images/rich_club.png')
    plt.close()

def degree_to_degree_correlation(G):
    # Calculate the average neighbor degree for each node in the smaller network
    avg_neighbor_deg_small = nx.average_neighbor_degree(G)

    # Calculate the degree of each node in the smaller network
    node_degrees = dict(G.degree())

    # Prepare data for plotting
    degrees = np.array(list(node_degrees.values()))
    avg_neighbors_deg = np.array(list(avg_neighbor_deg_small.values()))

    # Calculate degree assortativity coefficient of the smaller network
    assortativity_coefficient = nx.degree_assortativity_coefficient(G)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(degrees, avg_neighbors_deg, alpha=0.5, edgecolor='none')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Node Degree', fontsize=14)
    plt.ylabel('Average Neighbor Degree', fontsize=14)
    plt.title('Node Degree vs. Average Neighbor Degree', fontsize=16)
    plt.grid(True, which="both", ls="--")
    print(assortativity_coefficient)
    plt.savefig('./Images/node_degree.png')
    plt.close()

def simulate_epidemic(G, model='SIS', initial_infecteds=0.05, beta=0.9, gamma=0.01, tmax=50):
    """
    Simulate an epidemic (SIS or SIR model) on a network using EoN library.

    Parameters:
    - G: NetworkX graph
    - model: 'SIS' or 'SIR', the epidemic model to simulate
    - initial_infecteds: fraction or count of initially infected nodes
    - beta: infection rate
    - gamma: recovery rate
    - tmax: maximum simulation time

    Returns:
    - t: numpy array of times
    - S: numpy array of susceptible counts over time
    - I: numpy array of infected counts over time
    - R: numpy array of removed counts over time (only for SIR model)
    """
    N = G.number_of_nodes()
    initial_infecteds = int(N * initial_infecteds)
    if model == 'SIS':
        t, S, I = EoN.fast_SIS(G, tau=beta, gamma=gamma, tmax=tmax, 
                              initial_infecteds=initial_infecteds, return_full_data=False)
        save_simulation_results(t=t, S=S, I=I, filename_prefix = './Images/SIS/simulation_beta'+'{:.2f}'.format(beta)+'gamma_'+'{:.2f}'.format(gamma))
        return t, S, I, None
    elif model == 'SIR':
        t, S, I, R = EoN.fast_SIR(G, tau=beta, gamma=gamma, tmax=tmax, 
                                  initial_infecteds=initial_infecteds, return_full_data=False)
        save_simulation_results(t=t, S=S, I=I, R=R, filename_prefix = './Images/SIR/simulation_beta'+'{:.2f}'.format(beta)+'gamma_'+'{:.2f}'.format(gamma))
        return t, S, I, R

def simulate_intervention(G, strategy='random', fraction=0.1, beta=0.03, gamma=0.01, tmax=50):
    """
    Simulate an epidemic (SIS model) on a network with intervention strategies.

    Parameters:
    - G: NetworkX graph
    - strategy: 'random', 'targeted', or 'acquaintance', the intervention strategy
    - fraction: fraction of nodes to be removed/immunized
    - beta: infection rate
    - gamma: recovery rate
    - tmax: maximum simulation time

    Returns:
    - t: numpy array of times
    - S: numpy array of susceptible counts over time
    - I: numpy array of infected counts over time
    """
    N = G.number_of_nodes()
    num_to_remove = int(N * fraction)
    G_modified = G.copy()
    
    if strategy == 'random':
        nodes_to_remove = np.random.choice(G.nodes(), size=num_to_remove, replace=False)
    elif strategy == 'targeted':
        nodes_to_remove = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:num_to_remove]
    elif strategy == 'acquaintance':
        nodes_to_remove = set()
        while len(nodes_to_remove) < num_to_remove:
            node = np.random.choice(list(G.nodes()))
            if G.degree(node) > 0:
                neighbor = np.random.choice(list(G.neighbors(node)))
                nodes_to_remove.add(neighbor)
    
    G_modified.remove_nodes_from(nodes_to_remove)
    
    # Simulate SIS model on the modified graph
    t, S, I, _ = simulate_epidemic(G_modified, model='SIS', initial_infecteds=0.05, 
                                   beta=beta, gamma=gamma, tmax=tmax)
    save_simulation_results(t=t, S=S, I=I, filename_prefix = './Images/' + strategy + 'SIS_simulation')
    
    return t, S, I


def save_simulation_results(t, S, I, R=None, filename_prefix='./Images/simulation'):
    """
    Save simulation results to CSV and plot results.
    """
    # 将结果保存为DataFrame
    data = {'t': t, 'S': S, 'I': I}
    if R is not None:
        data['R'] = R
    df = pd.DataFrame(data)
    
    # 保存为CSV文件
    csv_filename = f'{filename_prefix}.csv'
    df.to_csv(csv_filename, index=False)
    print(f'Results saved to {csv_filename}')
    
    # 绘制结果并保存图片
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    if R is not None:
        plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Fraction of Population')
    plt.legend()
    figure_title = filename_prefix.split('/')[-1]
    plt.title(f'{figure_title} Results')
    
    # 保存图片
    img_filename = f'{filename_prefix}.png'
    plt.savefig(img_filename)
    plt.close()
    print(f'Results plot saved to {img_filename}')

def main():
    edge_list_path = './Twenty-Years-of-Network-Science/edge-list/twenty_years_edgelist.xlsx'
    G = load_network(file_path = edge_list_path)
    # rich_club(G)
    # degree_to_degree_correlation(G)
    for beta in np.arange(0.05, 1, 0.05):
        for gamma in np.arange(0.05, 1, 0.1):
            simulate_epidemic(G, beta = beta, gamma = gamma)
            simulate_epidemic(G = G, model = 'SIR', beta = beta, gamma = gamma)
    simulate_intervention(G)
    simulate_intervention(G, strategy = 'targeted')
    simulate_intervention(G, strategy = 'acquaintance')

if __name__ == '__main__':
    main()

