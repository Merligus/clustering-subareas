# run in clustering_subareas "python -m web.web"
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from algoritmos.smacof import MDS
from .data import Params, Data
import os
from algoritmos.finder import ClusterFinder
from flask import Flask, redirect, url_for, render_template, request, session, flash
from flask_session import Session
from flask_cors import CORS

app = Flask(__name__)
SECRET_KEY = "key para session"
SESSION_TYPE = 'filesystem'
app.secret_key = SECRET_KEY
app.session_type = SESSION_TYPE
app.config.from_object(__name__)
app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=False)
SERVER_NAME = '192.168.0.6:5000'
SESSION_COOKIE_DOMAIN = '192.168.0.6:5000'
# app.config['SERVER_NAME'] = SERVER_NAME
# app.config['SESSION_COOKIE_DOMAIN'] = SESSION_COOKIE_DOMAIN
Session(app)
CORS(app)
db = Data("_2010_only_journals", "union", "./data")

@app.route("/", methods=["POST", "GET"])
@app.route("/search", methods=["POST", "GET"])
def search():
    global db
    if "selected" not in session:
        session["selected"] = []
        session.modified = True
    
    if "in_name" not in session:
        session["in_name"] = "2010_only_journals"
        session.modified = True
    
    if "function" not in session:
        session["function"] = "agglomerative"
        session.modified = True

    if request.method == "POST" and "in_name" in request.form:
        session["in_name"] = request.form["in_name"]
        session.modified = True

    if request.method == "POST" and "function" in request.form:
        session["function"] = request.form["function"]
        session.modified = True
    
    new_in_name = f'_{session["in_name"]}'
    print(f'{db.in_name} vs {new_in_name}')
    if (db.in_name != new_in_name):
        db = Data(new_in_name, "union", "./data")

    if request.method == "POST" and "reset" in request.form:
        session["selected"] = []
        session.modified = True
        return render_template("search_venues.html", venues=db.journal_names_list, selected_l=session["selected"], in_name_=session["in_name"], function_=session["function"])
    elif request.method == "POST" and "sel" in request.form:
        if request.form["sel"] not in session["selected"]:
            session["selected"].append(request.form["sel"])
            session.modified = True
        return render_template("search_venues.html", venues=db.journal_names_list, selected_l=session["selected"], in_name_=session["in_name"], function_=session["function"])
    elif request.method == "POST" and "search" in request.form:
        selected_j = []
        for journal_selected in session["selected"]:
            for j, journal in db.journal_names_list:
                if journal_selected == journal:
                    selected_j.append(j)
        venues_s = set(selected_j)
        
        if len(venues_s) == 0:
            flash("Please provide a valid venue.", "info")
            return render_template("search_venues.html", venues=db.journal_names_list, selected_l=session["selected"], in_name_=session["in_name"], function_=session["function"])
            
        in_set_conf = set()
        s = ''
        for key in db.index_to_journalname:
            if db.index_to_journalname[key] in venues_s:
                in_set_conf.add(key)
                s += db.index_to_journalname[key] + ' '

        if len(s) == 0:
            print(f"Nenhum identificado")
            flash("Please provide a valid venue.", "error")
            return render_template("search_venues.html", venues=db.journal_names_list, selected_l=session["selected"], in_name_=session["in_name"], function_=session["function"])

        parsed = Params(request.form["in_name"], 'union', './data', request.form["function"], len(db.distance))
        
        parsed.old_cluster = in_set_conf
        parsed.cf = ClusterFinder(db.children[request.form["function"]], len(db.distance), in_set_conf, [])
        c, parsed.iteration = parsed.cf.find_cluster(0)
        if parsed.iteration == len(parsed.cf.children):
            print(f"Nenhum cluster achado")
            flash("No cluster found", "error")
            return render_template("search_venues.html", venues=db.journal_names_list, selected_l=session["selected"], in_name_=session["in_name"], function_=session["function"])

        parsed.cluster = parsed.cf.labels_sets[c]
        
        session["mode"] = "union"
        session["in_name"] = request.form["in_name"]
        session["function"] = request.form["function"]
        session["iteration"] = parsed.iteration
        session["in_set_conf"] = list(in_set_conf)
        session.modified = True

        for key in session:
            print(key, session[key])

        return redirect(url_for("listar_conferencias", next="0"))
    else:
        return render_template("search_venues.html", venues=db.journal_names_list, selected_l=session["selected"], in_name_=session["in_name"], function_=session["function"])

@app.route("/venues<next>", methods=["POST", "GET"])
def listar_conferencias(next):
    if "iteration" not in session:
        return redirect(url_for("search"))
        
    global db
    parsed = Params(session["in_name"], session["mode"], './data', session["function"], len(db.distance), session["iteration"], 
                    db.children[session["function"]], session["in_set_conf"])
    if request.method == "POST" or next == "1":
        parsed.old_cluster = parsed.cluster
        c, parsed.iteration = parsed.cf.find_cluster(parsed.iteration+1)
        if parsed.iteration < len(parsed.cf.children):
            parsed.cluster = parsed.cf.labels_sets[c]
            
        session["iteration"] = parsed.iteration
            
    i = 0
    old_cluster_l = []
    for vi in parsed.old_cluster:
        old_cluster_l.append((i, db.index_to_journal_complete_name[vi]))
        i += 1

    new_cluster = []
    for vi in parsed.cluster:
        if vi not in parsed.old_cluster:
            new_cluster.append((i, db.index_to_journal_complete_name[vi]))
            i += 1

    return render_template("show_venues.html", new=new_cluster, old=old_cluster_l, tam_l=len(parsed.cluster))

@app.route("/frequency")
def listar_frequencia():
    if "iteration" not in session:
        return redirect(url_for("search"))

    global db
    parsed = Params(session["in_name"], session["mode"], './data', session["function"], len(db.distance), session["iteration"], 
                    db.children[session["function"]], session["in_set_conf"])
    
    sentences = []
    for vi in parsed.cluster:
        sentences.append(db.index_to_journal_complete_name[vi].lower())
    lista = parsed.cf.show_top(sentences, n=10)

    return render_template("show_frequency.html", word_freq=lista)

@app.route("/graph")
def show_graph():
    if "iteration" not in session:
        return redirect(url_for("search"))

    global db
    parsed = Params(session["in_name"], session["mode"], './data', session["function"], len(db.distance), session["iteration"], 
                    db.children[session["function"]], session["in_set_conf"])

    if len(parsed.cluster) <= 2 or len(parsed.cluster) >= 15:
        flash("In order to show the graph, the cluster must have size more than 2 and less than 15", "error")
        return redirect(url_for("listar_conferencias", next="0"))
    
    g = nx.Graph()

    vertices = []
    for vi in parsed.cluster:
        vertices.append(vi)

    distance_temp = np.zeros((len(parsed.cluster), len(parsed.cluster)))
    m = MDS(ndim=2, weight_option="d-2", itmax=10000)

    journalname = {}
    node_size = []
    v1 = 0
    while v1 < len(parsed.cluster):
        v2 = v1 + 1
        while v2 < len(parsed.cluster):
            if db.distance[vertices[v1], vertices[v2]] > 0 and db.distance[vertices[v1], vertices[v2]] < np.inf:
                if db.adj_mat[vertices[v1], vertices[v2]] > 0:
                    g.add_edge(v1, v2, weight=db.adj_mat[vertices[v1], vertices[v2]])
                distance_temp[v1, v2] = distance_temp[v2, v1] = db.distance[vertices[v1], vertices[v2]]
            v2 += 1
        journalname[v1] = db.index_to_journalname[vertices[v1]]
        node_size.append(4*np.ceil(db.nauthors[vertices[v1]]/len(parsed.cluster)))
        v1 += 1

    mds_model = m.fit(distance_temp) # shape = journals x n_components
    X_transformed = mds_model['conf']

    width = nx.get_edge_attributes(g, 'weight')
    min_w = min(width.values())
    max_w = max(width.values())
    for w in width:
        width[w] = 0.5 + 4*(width[w] - min_w)/(max_w - min_w)

    edge_labels = {}
    for v1, v2, w in g.edges.data():
        edge_labels[(v1, v2)] = f"{db.adj_mat[vertices[v1], vertices[v2]]:.2f}" # w['weight']
        # print(f'{v1}:{journalname[v1]}:{nauthors[v1]}, {v2}:{journalname[v2]}:{nauthors[v2]} = {distance[vertices[v1], vertices[v2]]}:{w["weight"]}')

    fig = plt.figure(figsize=(24,24))
    ax = fig.add_axes([0,0,1,1])
    # pos = nx.spring_layout(g)
    pos = {}
    for vi in range(len(parsed.cluster)):
        pos[vi] = X_transformed[vi]
    # print(pos)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_size, node_color="#A8C1FB")
    nx.draw_networkx_labels(g, pos, ax=ax, labels=journalname, font_color="#DF0000", font_size=22)
    nx.draw_networkx_edges(g, pos, ax=ax, width=list(width.values()))
    nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=edge_labels, font_size=18)
    filename = f"graph{np.random.random()}.png"
    for file in os.listdir("./web/static/images/"):
        os.remove("./web/static/images/" + file)
    plt.savefig(f'./web/static/images/{filename}')

    del journalname
    del vertices
    del g

    # setting the list
    i = 0
    old_cluster_l = []
    for vi in parsed.old_cluster:
        old_cluster_l.append((i, db.index_to_journal_complete_name[vi]))
        i += 1

    new_cluster = []
    for vi in parsed.cluster:
        if vi not in parsed.old_cluster:
            new_cluster.append((i, db.index_to_journal_complete_name[vi]))
            i += 1
        
    return render_template("show_graph.html", src=filename, new=new_cluster, old=old_cluster_l, tam_l=len(parsed.cluster))

@app.route("/clean")
def clean():
    remove_l = []
    for key in session:
        remove_l.append(key)
    for key in remove_l:
        session.pop(key, None)
    return redirect(url_for("search"))

def run():
    app.run(debug=False)

run()