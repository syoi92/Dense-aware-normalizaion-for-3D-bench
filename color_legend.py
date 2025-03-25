import json
import matplotlib.pyplot as plt


def color_legend(class_json, output):

    # Extract names and colors
    names = [class_json[key]["name"] for key in class_json][::-1]
    colors = [class_json[key]["color"] for key in class_json][::-1]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, len(names) * 0.3))

    # Plot color legend
    for i, (name, color) in enumerate(zip(names, colors)):
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
        ax.text(1.1, i + 0.5, name, va='center', fontsize=12)

    # Formatting
    ax.set_xlim(0, 2)
    ax.set_ylim(0, len(names))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


    # Show legend
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.show()




def main():
    json_data = 'cnst_labell.json'
    output = './temp/color_legend.png'

    with open(json_data, 'r') as file:
        class_json = json.load(file)

    color_legend(class_json, output)


if __name__ == "__main__":
    main()



