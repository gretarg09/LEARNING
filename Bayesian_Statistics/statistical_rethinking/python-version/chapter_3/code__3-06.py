from models import globe_tossing_model


p_grid, posterior = globe_tossing_model(100)

# Add up posterior probability where p > 0.5
print(sum(posterior[p_grid < 0.5]))
