# uses networkD3 to render complex in json file

library(jsonlite)
library(networkD3)


file_path <- readline(prompt = 'path to json file that contains complex')
complex <- fromJSON(file_path)
Nodes <- data.frame('NodeID' = complex$vertex_IDs, 'Group' = complex$vertex_levels, 'Nodesize' = complex$vertex_sizes)
Links <- data.frame('Source' = complex$edge_sources, 'Target' = complex$edge_targets, 'Value' = complex$edge_values)

network <- forceNetwork(Nodes = Nodes, NodeID = 'NodeID',
                        Group = 'Group', Nodesize = 'Nodesize',
                        Links = Links, Source = 'Source',
                        Target = 'Target', Value = 'Value',
                        radiusCalculation = JS('Math.sqrt(d.nodesize)/2'),
                        opacity = 1, fontSize = 11,
                        linkDistance = 10, charge = -400)
print(network)
