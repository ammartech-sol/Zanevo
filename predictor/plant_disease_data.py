FINAL_CLASSES = [
    "citrus canker", "wheat stripe rust", "grape downy mildew", "apple scab",
    "soybean frog eye leaf spot", "bean rust", "zucchini powdery mildew",
    "corn smut", "tomato early blight", "cucumber angular leaf spot",
    "peach leaf curl", "banana black leaf streak", "coffee leaf rust",
    "cabbage black rot", "maple tar spot"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(FINAL_CLASSES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

DISEASE_ADVICE = {
    "citrus canker": {
        "what_it_is": "A bacterial disease caused by Xanthomonas citri, producing raised, corky brown lesions with a yellow halo on leaves, stems, and fruit of citrus trees.",
        "what_to_do": "Prune and destroy infected plant material, disinfect pruning tools between cuts, apply copper-based bactericides during wet, windy periods, and improve air circulation around trees.",
        "what_not_to_do": "Don't work on wet foliage, don't compost infected material, don't move infected plant material or tools between orchards, and don't ignore small lesions since the bacteria spreads quickly in rain and wind."
    },
    "wheat stripe rust": {
        "what_it_is": "A fungal disease (Puccinia striiformis) that produces yellow-orange powdery stripes running along the leaf veins, reducing photosynthesis and grain fill.",
        "what_to_do": "Plant resistant wheat varieties, apply fungicide early at first sign of stripes, monitor fields during cool, humid weather when the fungus spreads fastest, and rotate crops where possible.",
        "what_not_to_do": "Don't delay fungicide application once symptoms appear, don't rely on a single resistant variety across all seasons since rust strains evolve, and don't irrigate late in the day, which extends leaf wetness."
    },
    "grape downy mildew": {
        "what_it_is": "A disease caused by the oomycete Plasmopara viticola, appearing as yellow oily spots on the upper leaf surface and white downy growth underneath, especially after rain.",
        "what_to_do": "Apply protectant fungicides before rain events, improve canopy airflow through pruning and leaf removal, and remove fallen infected leaves at season's end to reduce overwintering spores.",
        "what_not_to_do": "Don't overhead irrigate, don't let canopies stay dense and humid, and don't wait for visible symptoms before your first preventive spray in high-risk wet seasons."
    },
    "apple scab": {
        "what_it_is": "A fungal disease (Venturia inaequalis) causing dark, scabby, olive-brown lesions on leaves and fruit, usually starting small and spreading with wet spring weather.",
        "what_to_do": "Apply fungicide starting at bud break through wet spring weather, rake and destroy fallen leaves in autumn to reduce spore sources, and prune for better air circulation.",
        "what_not_to_do": "Don't leave fallen leaves under the tree over winter, don't skip early-season sprays waiting for visible symptoms since scab is easiest to prevent before it establishes, and don't plant new trees in poorly ventilated, shaded spots."
    },
    "soybean frog eye leaf spot": {
        "what_it_is": "A fungal disease (Cercospora sojina) producing circular gray-brown spots with reddish-brown margins on soybean leaves, resembling a frog's eye.",
        "what_to_do": "Rotate crops away from soybean for at least a year, plant resistant varieties, apply foliar fungicide if disease pressure is high, and manage crop residue after harvest.",
        "what_not_to_do": "Don't plant soybean after soybean repeatedly in the same field, don't ignore early leaf spots in humid conditions, and don't skip residue management since the fungus survives on infected debris."
    },
    "bean rust": {
        "what_it_is": "A fungal disease (Uromyces appendiculatus) causing small reddish-brown pustules on the undersides of bean leaves, which can cause leaves to yellow and drop in severe cases.",
        "what_to_do": "Plant resistant varieties, apply fungicide at first sign of pustules, avoid overhead watering, and remove volunteer bean plants that can carry the fungus between seasons.",
        "what_not_to_do": "Don't water in the evening since prolonged leaf wetness favors spore germination, don't crowd plants too closely, and don't compost infected bean debris."
    },
    "zucchini powdery mildew": {
        "what_it_is": "A fungal disease appearing as white powdery patches on the surface of leaves and stems, thriving in warm, dry conditions with high humidity at night.",
        "what_to_do": "Apply sulfur or potassium bicarbonate-based fungicides at first sign, space plants for airflow, and choose resistant varieties where available.",
        "what_not_to_do": "Don't over-fertilize with nitrogen, which encourages dense, susceptible growth, don't let plants sit under dense shade with poor airflow, and don't wait until mildew has spread across most of the leaf to treat it."
    },
    "corn smut": {
        "what_it_is": "A fungal disease (Ustilago maydis) causing large, gray-white galls filled with black spores on ears, stalks, and tassels of corn.",
        "what_to_do": "Remove and destroy galls before they rupture and release spores, rotate crops, and avoid mechanical injury to plants during cultivation, since wounds are common infection points.",
        "what_not_to_do": "Don't leave ruptured galls in the field, don't over-apply nitrogen fertilizer which increases susceptibility, and don't till infected debris back into soil where spores can persist for years."
    },
    "tomato early blight": {
        "what_it_is": "A fungal disease (Alternaria solani) causing dark, concentric-ringed 'target' spots on lower, older leaves first, which then spread upward.",
        "what_to_do": "Remove infected lower leaves promptly, mulch around the base to prevent soil splash onto leaves, apply fungicide preventively in humid conditions, and rotate crops away from tomato and related nightshades.",
        "what_not_to_do": "Don't water from overhead, don't work with plants when foliage is wet, and don't leave infected debris in the garden over winter."
    },
    "cucumber angular leaf spot": {
        "what_it_is": "A bacterial disease (Pseudomonas syringae pv. lachrymans) causing angular, water-soaked spots bound by leaf veins, which can turn brown and drop out, leaving ragged holes.",
        "what_to_do": "Use disease-free seed, apply copper-based bactericides preventively, avoid working in wet fields, and rotate crops away from cucurbits for at least a year.",
        "what_not_to_do": "Don't handle plants when foliage is wet since this spreads bacteria on hands and tools, don't overhead irrigate, and don't save seed from infected plants."
    },
    "peach leaf curl": {
        "what_it_is": "A fungal disease (Taphrina deformans) causing leaves to pucker, thicken, and curl, often turning red or yellow, appearing in early spring as leaves emerge.",
        "what_to_do": "Apply a dormant fungicide spray in late fall or winter before buds swell, which is the only truly effective timing, and remove severely infected leaves during the growing season.",
        "what_not_to_do": "Don't wait until curled leaves appear in spring to spray, since by then the infection is already established and treatment won't reverse it, that window closes once buds open."
    },
    "banana black leaf streak": {
        "what_it_is": "Also called black Sigatoka, a fungal disease (Pseudocercospora fijiensis) causing dark brown to black streaks on leaves that merge into large necrotic patches, reducing photosynthesis and fruit yield.",
        "what_to_do": "Remove and destroy heavily infected leaves, apply fungicide on a rotation schedule to prevent resistance buildup, and improve field drainage and plant spacing for airflow.",
        "what_not_to_do": "Don't rely on a single fungicide repeatedly since the fungus develops resistance quickly, don't plant too densely, and don't ignore early streaking since it spreads rapidly in humid climates."
    },
    "coffee leaf rust": {
        "what_it_is": "A fungal disease (Hemileia vastatrix) producing orange-yellow powdery spots on the undersides of coffee leaves, leading to leaf drop and reduced yields in severe outbreaks.",
        "what_to_do": "Plant rust-resistant coffee varieties, apply copper-based or systemic fungicides preventively before rainy seasons, and prune for better airflow and sunlight penetration.",
        "what_not_to_do": "Don't over-shade coffee plants excessively, don't delay fungicide application until after visible defoliation, and don't neglect plant nutrition since stressed plants are more susceptible."
    },
    "cabbage black rot": {
        "what_it_is": "A bacterial disease (Xanthomonas campestris pv. campestris) causing yellow V-shaped lesions starting at leaf edges that progress inward, with blackened veins.",
        "what_to_do": "Use certified disease-free seed or transplants, rotate crops away from brassicas for at least two years, and remove and destroy infected plants promptly.",
        "what_not_to_do": "Don't work in fields when plants are wet, don't compost infected plant material, and don't replant brassicas in the same soil without a proper rotation break."
    },
    "maple tar spot": {
        "what_it_is": "A fungal disease (Rhytisma species) causing distinctive black, tar-like spots on maple leaves, mostly a cosmetic issue rather than a serious threat to tree health.",
        "what_to_do": "Rake and dispose of fallen leaves in autumn to reduce next season's spore load, and maintain overall tree health through proper watering and fertilization.",
        "what_not_to_do": "Don't be alarmed into aggressive fungicide treatment for what is typically a cosmetic condition, and don't leave infected leaf litter under the tree over winter."
    }
}