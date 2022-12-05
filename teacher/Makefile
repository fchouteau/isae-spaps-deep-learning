build-student-notebooks:
	jupytext --to notebook 1_deep_learning.py --output ../1_deep_learning.ipynb
	jupytext --to notebook 2_image_convolutions_cnns.py --output ../2_image_convolutions_cnns.ipynb
	jupytext --to notebook 3_aircraft_classifier.py --output ../3_aircraft_classifier.ipynb
	jupytext --to notebook 4_sliding_window.py --output ../4_sliding_window.ipynb

	jupyter nbconvert --to notebook \
	    --TagRemovePreprocessor.enabled=True \
	    --TagRemovePreprocessor.remove_cell_tags=solution \
	    --inplace \
        ../1_deep_learning.ipynb

	jupyter nbconvert --to notebook \
	    --TagRemovePreprocessor.enabled=True \
	    --TagRemovePreprocessor.remove_cell_tags=solution \
	    --inplace \
        ../2_image_convolutions_cnns.ipynb

	jupyter nbconvert --to notebook \
	    --TagRemovePreprocessor.enabled=True \
	    --TagRemovePreprocessor.remove_cell_tags=solution \
	    --inplace \
        ../3_aircraft_classifier.ipynb

	jupyter nbconvert --to notebook \
	    --TagRemovePreprocessor.enabled=True \
	    --TagRemovePreprocessor.remove_cell_tags=solution \
	    --inplace \
        ../4_sliding_window.ipynb
